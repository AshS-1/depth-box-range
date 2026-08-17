"""Deprojection, RANSAC plane fitting, and plane-constrained box fitting.

Coordinate convention throughout is the *depth optical frame*: +x right,
+y down, +z forward along the optical axis. This matches OpenCV and RealSense.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .intrinsics import CameraIntrinsics


def deproject(depth_m: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """Depth image (H, W) in metres -> organised point cloud (H, W, 3).

    Invalid pixels (depth <= 0) come back as NaN so they propagate rather than
    landing at the origin and corrupting every downstream fit.
    """
    h, w = depth_m.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z = depth_m.astype(np.float32, copy=True)
    z[z <= 0] = np.nan

    x = (uu - intr.cx) * z / intr.fx
    y = (vv - intr.cy) * z / intr.fy
    return np.stack((x, y, z), axis=-1)


@dataclass(frozen=True)
class Plane:
    """Plane ``n . p + d = 0`` with unit normal ``n``."""

    normal: np.ndarray
    d: float
    inlier_count: int = 0
    rms: float = 0.0

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.d

    @property
    def point(self) -> np.ndarray:
        """The point on the plane closest to the camera origin."""
        return -self.d * self.normal

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Two orthonormal vectors spanning the plane."""
        seed = np.array([1.0, 0.0, 0.0])
        if abs(self.normal @ seed) > 0.9:
            seed = np.array([0.0, 0.0, 1.0])
        e1 = np.cross(self.normal, seed)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.normal, e1)
        return e1, e2


def fit_plane_ransac(
    points: np.ndarray,
    *,
    threshold: float = 0.015,
    iterations: int = 200,
    up_hint: np.ndarray | None = None,
    max_tilt_deg: float = 45.0,
    rng: np.random.Generator | None = None,
) -> Plane | None:
    """Fit the dominant plane to an (N, 3) array of finite points.

    ``up_hint`` optionally constrains the result: candidate planes whose normal
    deviates from the hint by more than ``max_tilt_deg`` are discarded. That
    stops a large wall or the side of the box itself from winning the vote when
    what you actually want is the floor.

    All ``iterations`` hypotheses are scored in one vectorised pass.
    """
    if len(points) < 3:
        return None
    rng = rng or np.random.default_rng(0)

    idx = rng.integers(0, len(points), size=(iterations, 3))
    tri = points[idx]  # (iters, 3, 3)

    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-9
    if not valid.any():
        return None
    normals[valid] /= norms[valid, None]

    ds = -np.einsum("ij,ij->i", normals, tri[:, 0])

    if up_hint is not None:
        up = np.asarray(up_hint, dtype=np.float64)
        up = up / np.linalg.norm(up)
        # Sign-agnostic: a plane is the same plane with the normal flipped.
        valid &= np.abs(normals @ up) >= np.cos(np.deg2rad(max_tilt_deg))
    if not valid.any():
        return None

    # (iters, N) residuals would blow up memory on a full cloud, so subsample
    # the scoring set. 20k points is plenty to rank hypotheses.
    if len(points) > 20000:
        score_pts = points[rng.choice(len(points), 20000, replace=False)]
    else:
        score_pts = points

    resid = np.abs(normals @ score_pts.T + ds[:, None])
    counts = np.where(valid, (resid < threshold).sum(axis=1), -1)
    best = int(np.argmax(counts))
    if counts[best] <= 0:
        return None

    # Refine on the winner's inliers via least squares (PCA normal).
    inlier_mask = np.abs(points @ normals[best] + ds[best]) < threshold
    inliers = points[inlier_mask]
    if len(inliers) < 3:
        return None

    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    n = vt[-1]
    n /= np.linalg.norm(n)
    if up_hint is not None and n @ np.asarray(up_hint, dtype=np.float64) < 0:
        n = -n
    d = -float(n @ centroid)

    resid_final = np.abs(inliers @ n + d)
    return Plane(n, d, int(inlier_mask.sum()), float(np.sqrt((resid_final**2).mean())))


@dataclass(frozen=True)
class OrientedBox:
    """An oriented box: ``center`` with axes ``R`` (columns) and full ``extents``."""

    center: np.ndarray
    R: np.ndarray
    extents: np.ndarray

    def corners(self) -> np.ndarray:
        """The 8 corners, (8, 3)."""
        signs = np.array(
            [
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
            ],
            dtype=np.float64,
        )
        return self.center + (signs * self.extents / 2.0) @ self.R.T

    @property
    def yaw(self) -> float:
        """Rotation of the box's first horizontal axis about the plane normal."""
        return float(np.arctan2(self.R[1, 0], self.R[0, 0]))


def fit_box_on_plane(
    points: np.ndarray, plane: Plane, *, trim_percent: float = 2.0
) -> OrientedBox | None:
    """Fit an oriented box to ``points``, constrained to stand on ``plane``.

    A free 3D PCA fit is the obvious approach and it is the wrong one here: a
    depth camera only ever sees one or two faces of a box, so the point mass is
    a hollow L-shell and the principal axes tilt away from the true edges.

    Constraining one axis to the ground normal removes that failure mode. The
    remaining two axes are found with ``cv2.minAreaRect`` on the footprint --
    the points projected down onto the plane -- which recovers the true edge
    directions from a visible corner even though the back of the box is missing.
    """
    if len(points) < 10:
        return None

    e1, e2 = plane.basis()
    origin = plane.point
    rel = points - origin

    uv = np.stack((rel @ e1, rel @ e2), axis=1).astype(np.float32)
    height = plane.signed_distance(points)

    rect = cv2.minAreaRect(uv)
    quad = cv2.boxPoints(rect).astype(np.float64)  # (4, 2), ordered around the rect

    # Derive the axes from the corners rather than the angle: OpenCV's angle
    # convention has changed between releases, the corner order has not.
    edge_a = quad[1] - quad[0]
    edge_b = quad[2] - quad[1]
    if np.linalg.norm(edge_a) < 1e-6 or np.linalg.norm(edge_b) < 1e-6:
        return None
    axis_a_2d = edge_a / np.linalg.norm(edge_a)
    axis_b_2d = edge_b / np.linalg.norm(edge_b)

    # Take only the *directions* from minAreaRect and re-measure the lengths by
    # percentile. minAreaRect is a hull operation, so every noisy outlier pushes
    # the rectangle outward and never inward -- the error is one-sided, and it
    # grows with the z^2 depth noise. At 3.5 m that inflated a 0.40 m box to
    # 0.62 m. Trimming the tails costs a hair of true extent and removes the bias.
    proj_a = uv.astype(np.float64) @ axis_a_2d
    proj_b = uv.astype(np.float64) @ axis_b_2d
    lo_a, hi_a = np.percentile(proj_a, [trim_percent, 100.0 - trim_percent])
    lo_b, hi_b = np.percentile(proj_b, [trim_percent, 100.0 - trim_percent])

    len_a = float(hi_a - lo_a)
    len_b = float(hi_b - lo_b)
    if len_a < 1e-6 or len_b < 1e-6:
        return None

    a1 = axis_a_2d[0] * e1 + axis_a_2d[1] * e2
    a2 = axis_b_2d[0] * e1 + axis_b_2d[1] * e2
    n = plane.normal

    # The box sits on the plane, so it spans from the ground up to its top face.
    # Use a high percentile rather than the max to shrug off flyer pixels.
    top = float(np.percentile(height, 100.0 - trim_percent))
    top = max(top, 1e-3)

    # Centre is the midpoint of the trimmed spans, consistent with the extents.
    center_2d = ((lo_a + hi_a) / 2.0) * axis_a_2d + ((lo_b + hi_b) / 2.0) * axis_b_2d
    center = (
        origin
        + center_2d[0] * e1
        + center_2d[1] * e2
        + (top / 2.0) * n
    )

    R = np.stack((a1, a2, n), axis=1)
    # Keep it a proper right-handed rotation.
    if np.linalg.det(R) < 0:
        R[:, 1] = -R[:, 1]

    return OrientedBox(center, R, np.array([len_a, len_b, top], dtype=np.float64))


def face_tolerance(box: OrientedBox, sigma_m: float, *, floor_m: float = 0.02) -> float:
    """How close a point must be to a face plane to count as lying on it.

    Scales with the sensor noise at this range, but is capped at a fraction of
    the box's own smallest half-extent -- without that cap, a far-away small box
    gets a tolerance wide enough that a single point counts as being on two
    opposite faces at once, and the count becomes meaningless in the other
    direction.
    """
    cap = 0.25 * float(np.min(box.extents)) / 2.0
    return float(np.clip(1.5 * sigma_m, floor_m, max(cap, floor_m)))


def count_visible_faces(
    points: np.ndarray,
    box: OrientedBox,
    *,
    tol: float = 0.02,
    min_frac: float = 0.08,
    exclude_bottom: bool = True,
) -> int:
    """How many of the box's faces carry a meaningful share of the points.

    Drives the confidence score: with only one face visible the box's depth
    along the viewing direction is unobservable, so the centre estimate is a
    guess and the caller deserves to know.

    ``tol`` must track the sensor's noise at the box's range or this silently
    becomes a distance meter instead of a geometry check -- at 3.5 m the depth
    scatter alone is ~5 cm, so a fixed 2 cm band reports faces vanishing when
    nothing about the view changed. See :func:`face_tolerance`.
    """
    if len(points) == 0:
        return 0
    local = (points - box.center) @ box.R  # (N, 3) in box axes
    half = box.extents / 2.0

    seen = 0
    for axis in range(3):
        for sign in (-1.0, 1.0):
            # Axis 2 is the plane normal by construction, so (2, -1) is the face
            # resting on the ground. It is never observable, but points along
            # the bottom edge of a *side* face sit within tol of it, which used
            # to push the count to 4 -- impossible for a convex box, which shows
            # at most 3 faces from any one viewpoint.
            if exclude_bottom and axis == 2 and sign < 0:
                continue
            on_face = np.abs(local[:, axis] - sign * half[axis]) < tol
            if on_face.mean() >= min_frac:
                seen += 1
    return min(seen, 3)
