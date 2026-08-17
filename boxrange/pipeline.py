"""End-to-end: depth frame in, tracked box distances out.

The per-frame chain is segment -> fit plane-constrained box -> range. On top of
that sits a small tracker, which is not optional polish: a single frame's range
carries the full z^2 stereo noise, and a box that is genuinely static should not
have its reported distance jitter by centimetres. Associating detections across
frames and smoothing lets the estimate settle without adding lag to real motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import Frame
from .geometry import OrientedBox, Plane, fit_box_on_plane
from .ranging import RangeEstimate, estimate_range
from .segment import Candidate, PlaneClusterDetector


@dataclass(frozen=True)
class BoxDetection:
    """One box found in one frame."""

    box: OrientedBox
    range: RangeEstimate
    candidate: Candidate
    plane: Plane
    track_id: int = -1
    smoothed_surface_m: float = float("nan")

    @property
    def extents_m(self) -> np.ndarray:
        return self.box.extents


@dataclass
class _Track:
    track_id: int
    centroid: np.ndarray
    surface_m: float
    hits: int = 1
    misses: int = 0

    def update(self, centroid: np.ndarray, surface_m: float, alpha: float) -> None:
        self.centroid = (1 - alpha) * self.centroid + alpha * centroid
        self.surface_m = (1 - alpha) * self.surface_m + alpha * surface_m
        self.hits += 1
        self.misses = 0


@dataclass
class BoxRangePipeline:
    """Stateful pipeline. Feed it frames in order.

    ``max_boxes`` caps work per frame -- the candidates arrive sorted by size,
    and past the first few they are noise.
    """

    detector: PlaneClusterDetector = field(default_factory=PlaneClusterDetector)
    max_boxes: int = 3
    min_confidence: float = 0.05
    smoothing: float = 0.35
    gate_m: float = 0.30
    max_misses: int = 5

    _tracks: list[_Track] = field(default_factory=list, init=False)
    _next_id: int = field(default=0, init=False)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    def process(self, frame: Frame) -> list[BoxDetection]:
        candidates, plane = self.detector.detect(
            frame.depth_m, frame.intrinsics, frame.color
        )
        if plane is None:
            self._age_tracks(set())
            return []

        detections: list[BoxDetection] = []
        for cand in candidates[: self.max_boxes]:
            box = fit_box_on_plane(cand.points, plane)
            if box is None:
                continue
            try:
                rng = estimate_range(cand.points, box, frame.intrinsics, plane=plane)
            except ValueError:
                continue
            if rng.confidence < self.min_confidence:
                continue
            detections.append(
                BoxDetection(box=box, range=rng, candidate=cand, plane=plane)
            )

        return self._track(detections)

    # -- tracking ---------------------------------------------------------

    def _track(self, detections: list[BoxDetection]) -> list[BoxDetection]:
        """Greedy nearest-centroid association, closest pairs first.

        Greedy is the right complexity here: with a handful of boxes the
        optimal assignment and the greedy one agree, and the gate does the real
        work of preventing identity swaps.
        """
        out: list[BoxDetection] = []
        used_tracks: set[int] = set()

        pairs = []
        for di, det in enumerate(detections):
            c = det.candidate.centroid
            for ti, track in enumerate(self._tracks):
                dist = float(np.linalg.norm(track.centroid - c))
                if dist < self.gate_m:
                    pairs.append((dist, di, ti))
        pairs.sort()

        det_to_track: dict[int, int] = {}
        for _, di, ti in pairs:
            if di in det_to_track or ti in used_tracks:
                continue
            det_to_track[di] = ti
            used_tracks.add(ti)

        for di, det in enumerate(detections):
            if di in det_to_track:
                track = self._tracks[det_to_track[di]]
                track.update(det.candidate.centroid, det.range.surface_m, self.smoothing)
            else:
                track = _Track(
                    track_id=self._next_id,
                    centroid=det.candidate.centroid,
                    surface_m=det.range.surface_m,
                )
                self._next_id += 1
                self._tracks.append(track)
                used_tracks.add(len(self._tracks) - 1)

            out.append(
                BoxDetection(
                    box=det.box,
                    range=det.range,
                    candidate=det.candidate,
                    plane=det.plane,
                    track_id=track.track_id,
                    smoothed_surface_m=track.surface_m,
                )
            )

        self._age_tracks(used_tracks)
        return out

    def _age_tracks(self, used: set[int]) -> None:
        for i, track in enumerate(self._tracks):
            if i not in used:
                track.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]


def detection_to_dict(det: BoxDetection, frame: Frame) -> dict:
    """Flat JSON-safe record, for logging or piping into another process."""
    r = det.range
    return {
        "frame": frame.index,
        "timestamp_s": round(frame.timestamp_s, 4),
        "track_id": det.track_id,
        "distance_m": round(r.surface_m, 4),
        "distance_smoothed_m": round(det.smoothed_surface_m, 4),
        "axial_m": round(r.axial_m, 4),
        "centroid_m": round(r.centroid_m, 4),
        "measured_m": round(r.measured_m, 4),
        "sigma_m": round(r.sigma_m, 4),
        "confidence": round(r.confidence, 3),
        "visible_faces": r.visible_faces,
        "fit_rms_m": round(r.fit_rms_m, 4),
        "n_points": r.n_points,
        "extents_m": [round(float(e), 4) for e in det.box.extents],
        "center_xyz_m": [round(float(v), 4) for v in det.box.center],
        "yaw_deg": round(float(np.rad2deg(det.box.yaw)), 2),
    }
