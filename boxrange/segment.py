"""Isolating box-shaped candidates from a depth frame.

The geometry module fits a box to points *it is given*. This module decides
which points those are, and it is the stage a learned detector would replace:
:class:`Detector` is the seam. A region prior from EfficientPose or
FoundationPose can supply the mask; the depth fit still supplies the metric.

The approach here is subtract-and-cluster: find the dominant support plane,
drop it, then split what is left into connected surfaces. Clustering runs on
the *organised* cloud rather than as a generic 3D Euclidean clustering, because
the image grid already encodes adjacency -- two pixels are neighbours unless
depth jumps between them. That turns an O(N log N) spatial-index problem into
a connected-components pass over a mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .geometry import Plane, deproject, fit_plane_ransac
from .intrinsics import CameraIntrinsics


@dataclass(frozen=True)
class Candidate:
    """A cluster of points that might be a box."""

    points: np.ndarray  # (N, 3) finite points in the depth optical frame
    mask: np.ndarray  # (H, W) bool, which pixels produced them
    label: int

    @property
    def size(self) -> int:
        return len(self.points)

    @property
    def centroid(self) -> np.ndarray:
        return self.points.mean(axis=0)


class Detector(Protocol):
    """Produces box candidates from a depth frame.

    Implement this to swap in a learned front end. The contract is deliberately
    narrow: return candidate point sets, and let the geometric stage own the
    metric estimate.
    """

    def detect(
        self, depth_m: np.ndarray, intrinsics: CameraIntrinsics, color: np.ndarray | None
    ) -> tuple[list[Candidate], Plane | None]: ...


def depth_discontinuities(depth_m: np.ndarray, tol: np.ndarray) -> np.ndarray:
    """Pixels whose depth jumps by more than ``tol`` from a 4-neighbour.

    ``tol`` is per-pixel rather than a constant because the acceptable jump at
    4 m is far larger than at 0.5 m -- a fixed threshold either shreds distant
    surfaces or merges near ones into the wall behind them.
    """
    h, w = depth_m.shape
    z = np.where(depth_m > 0, depth_m, np.nan)
    pad = np.pad(z, 1, mode="constant", constant_values=np.nan)

    worst = np.zeros((h, w), dtype=np.float32)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        jump = np.abs(nb - z)
        worst = np.fmax(worst, np.nan_to_num(jump, nan=0.0))
    return worst > tol


@dataclass
class PlaneClusterDetector:
    """Default detector: remove the support plane, cluster the rest.

    ``min_height_m`` sets how far above the floor a point must sit to count,
    which is what stops floor noise from becoming a phantom box. ``max_extent_m``
    discards clusters too large to be a box -- walls, mostly.
    """

    plane_threshold_m: float = 0.015
    plane_iterations: int = 300
    min_height_m: float = 0.03
    max_height_m: float = 2.5
    min_points: int = 400
    max_extent_m: float = 2.0
    jump_sigmas: float = 6.0
    jump_floor_m: float = 0.012
    max_range_m: float = 6.0
    up_hint: tuple[float, float, float] = (0.0, -1.0, 0.0)
    max_tilt_deg: float = 45.0
    seed: int = 0

    def detect(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        color: np.ndarray | None = None,
    ) -> tuple[list[Candidate], Plane | None]:
        rng = np.random.default_rng(self.seed)
        cloud = deproject(depth_m, intrinsics)
        finite = np.isfinite(cloud[..., 2]) & (cloud[..., 2] < self.max_range_m)
        if finite.sum() < 3:
            return [], None

        plane = fit_plane_ransac(
            cloud[finite],
            threshold=self.plane_threshold_m,
            iterations=self.plane_iterations,
            up_hint=np.asarray(self.up_hint, dtype=np.float64),
            max_tilt_deg=self.max_tilt_deg,
            rng=rng,
        )
        if plane is None:
            return [], None

        # Height above the support plane. NaNs stay NaN and fail every compare.
        height = np.full(depth_m.shape, np.nan, dtype=np.float64)
        height[finite] = plane.signed_distance(cloud[finite])

        above = finite & (height > self.min_height_m) & (height < self.max_height_m)
        if above.sum() < self.min_points:
            return [], plane

        tol = self.jump_sigmas * intrinsics.range_sigma(
            np.where(depth_m > 0, depth_m, 1.0)
        ) + self.jump_floor_m
        keep = above & ~depth_discontinuities(depth_m, tol.astype(np.float32))

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            keep.astype(np.uint8), connectivity=8
        )

        # Rank by the areas OpenCV already computed. Testing `labels == i` for
        # every label costs a full-image pass each time, and dropout noise
        # produces ~1000 tiny components per frame -- that loop alone was 60% of
        # the frame budget. Only labels that clear min_points get a mask built.
        areas = stats[1:, cv2.CC_STAT_AREA]
        keepers = np.nonzero(areas >= self.min_points)[0] + 1

        candidates: list[Candidate] = []
        for label in keepers[np.argsort(-areas[keepers - 1])]:
            mask = labels == label
            pts = cloud[mask]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) < self.min_points:
                continue
            # Reject anything with a footprint no box would have.
            span = pts.max(axis=0) - pts.min(axis=0)
            if float(np.linalg.norm(span)) > self.max_extent_m * np.sqrt(3.0):
                continue
            candidates.append(Candidate(points=pts, mask=mask, label=int(label)))

        return candidates, plane


@dataclass
class RegionPriorDetector:
    """Adapter for a learned 2D detector.

    ``regions`` are pixel masks (or boxes converted to masks) from any RGB
    model -- EfficientPose's 2D head, a YOLO box, SAM, whatever. The support
    plane is still fitted from depth so the box fit stays plane-constrained,
    and each region is intersected with the above-plane mask so background
    bleeding inside a loose 2D box does not poison the fit.

    This is the intended integration point for the RGB pose networks: they
    localise, depth measures.
    """

    inner: PlaneClusterDetector

    def detect_with_regions(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        regions: list[np.ndarray],
    ) -> tuple[list[Candidate], Plane | None]:
        cands, plane = self.inner.detect(depth_m, intrinsics)
        if plane is None or not regions:
            return cands, plane

        out: list[Candidate] = []
        for i, region in enumerate(regions):
            merged = np.zeros_like(region, dtype=bool)
            for c in cands:
                # Keep a cluster if most of it falls inside the region.
                overlap = (c.mask & region).sum()
                if overlap > 0.5 * c.mask.sum():
                    merged |= c.mask
            if not merged.any():
                continue
            cloud = deproject(depth_m, intrinsics)
            pts = cloud[merged]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) >= self.inner.min_points:
                out.append(Candidate(points=pts, mask=merged, label=i + 1))
        return out, plane
