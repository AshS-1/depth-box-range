"""Turning a fitted box into a distance, with an honest error bar.

"Distance to the box" is ambiguous, and picking the wrong definition is a real
source of bugs. Four are reported:

``surface_m``    Euclidean camera-to-nearest-point-on-the-box. The number you
                 want for collision and stopping distance.
``axial_m``      Nearest box surface measured along +z. The number you want for
                 a forward-driving robot, since lateral offset should not
                 inflate "how far ahead is it".
``centroid_m``   Euclidean to the box centre. The number you want for grasping
                 or for handing a target to a planner.
``measured_m``   Robust nearest observed depth, model-free. Cross-check: if this
                 disagrees badly with ``surface_m``, the fit is wrong.

The uncertainty is not decoration. Stereo range error grows as z^2, so a box at
4 m carries roughly sixteen times the variance it does at 1 m, and a consumer
of a bare float has no way to know that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import OrientedBox, Plane, count_visible_faces, face_tolerance
from .intrinsics import CameraIntrinsics


@dataclass(frozen=True)
class RangeEstimate:
    """Distances in metres, plus how much to trust them."""

    surface_m: float
    axial_m: float
    centroid_m: float
    measured_m: float
    sigma_m: float
    confidence: float
    visible_faces: int
    fit_rms_m: float
    n_points: int

    @property
    def interval_95(self) -> tuple[float, float]:
        """Approximate 95% interval on ``surface_m``."""
        return (self.surface_m - 1.96 * self.sigma_m, self.surface_m + 1.96 * self.sigma_m)

    def __str__(self) -> str:
        return (
            f"{self.surface_m:.3f} m +/- {self.sigma_m:.3f} "
            f"(axial {self.axial_m:.3f}, centre {self.centroid_m:.3f}, "
            f"faces {self.visible_faces}, conf {self.confidence:.2f})"
        )


def closest_point_on_box(box: OrientedBox, query: np.ndarray | None = None) -> np.ndarray:
    """Point on the box's surface nearest ``query`` (default: camera origin).

    Clamping into the box's own frame gives the exact answer for an oriented
    box, which beats sampling corners or faces.
    """
    q = np.zeros(3) if query is None else np.asarray(query, dtype=np.float64)
    local = box.R.T @ (q - box.center)
    half = box.extents / 2.0
    return box.center + box.R @ np.clip(local, -half, half)


def box_surface_residuals(points: np.ndarray, box: OrientedBox) -> np.ndarray:
    """Per-point distance to the nearest face plane of ``box``.

    Points on a well-fitted box sit on some face, so this is small for all of
    them. It is the fit-quality signal that feeds confidence.
    """
    local = np.abs((points - box.center) @ box.R)
    half = box.extents / 2.0
    return np.min(np.abs(local - half), axis=1)


def _confidence(
    visible_faces: int,
    fit_rms: float,
    n_points: int,
    plane: Plane | None,
    sigma_expected: float,
) -> float:
    """Blend the things that actually predict whether the fit is trustworthy.

    Every term is scored *relative to what this sensor can do at this range*,
    never against an absolute millimetre threshold. A 5.7 cm residual is a
    terrible fit at 1 m and a textbook-perfect one at 3.5 m, where the depth
    noise floor is itself 5 cm. Scoring absolutely makes confidence a proxy for
    distance and quietly drops every far detection.

    Visible faces dominates: with one face visible the box's extent along the
    viewing direction is geometrically unobservable, so the centre is inferred
    rather than measured, and the caller should know the number is softer than
    its precision suggests.
    """
    face_term = {0: 0.10, 1: 0.45, 2: 1.00, 3: 1.00}.get(visible_faces, 1.0)

    # Residual in units of the expected noise. At or below the noise floor the
    # fit is as good as the hardware allows; 5x the floor means a bad fit.
    floor = max(sigma_expected, 0.003)
    ratio = fit_rms / floor
    rms_term = float(np.clip(1.0 - (ratio - 1.5) / 3.5, 0.15, 1.0))

    # Saturating: a box at range legitimately yields ~1k pixels, and the sigma
    # already accounts for how much averaging that buys.
    count_term = float(np.clip(n_points / 1500.0, 0.25, 1.0))
    plane_term = 1.0 if plane is None else float(np.clip(1.0 - plane.rms / 0.03, 0.3, 1.0))

    return float(np.clip(face_term * rms_term * count_term * plane_term, 0.0, 1.0))


def estimate_range(
    points: np.ndarray,
    box: OrientedBox,
    intrinsics: CameraIntrinsics,
    *,
    plane: Plane | None = None,
    percentile: float = 1.0,
) -> RangeEstimate:
    """Distances to ``box``, given the ``points`` it was fitted from.

    ``measured_m`` uses a low percentile of observed depth rather than the
    outright minimum: stereo matching produces occasional flyers that land
    metres in front of the true surface, and a single such pixel would drag a
    raw ``min()`` toward the camera -- exactly the failure the noise model in
    :mod:`intrinsics` predicts.
    """
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        raise ValueError("no finite points to range")

    near = closest_point_on_box(box)
    surface = float(np.linalg.norm(near))

    corners = box.corners()
    axial = float(corners[:, 2].min())
    centroid = float(np.linalg.norm(box.center))
    measured = float(np.percentile(pts[:, 2], percentile))

    sigma_expected = float(intrinsics.range_sigma(max(surface, 1e-3)))

    residuals = box_surface_residuals(pts, box)
    fit_rms = float(np.sqrt((residuals**2).mean()))
    faces = count_visible_faces(pts, box, tol=face_tolerance(box, sigma_expected))

    # Random per-pixel error averages down over the surface; systematic error
    # (quantisation, calibration, and a biased fit) does not. Keep both, and cap
    # the averaging benefit so a huge cluster cannot claim absurd precision.
    n_eff = float(np.clip(len(pts), 1, 5000))
    sigma_random = sigma_expected / np.sqrt(n_eff)
    sigma = float(np.sqrt(sigma_random**2 + fit_rms**2 + (intrinsics.depth_scale) ** 2))

    return RangeEstimate(
        surface_m=surface,
        axial_m=axial,
        centroid_m=centroid,
        measured_m=measured,
        sigma_m=sigma,
        confidence=_confidence(faces, fit_rms, len(pts), plane, sigma_expected),
        visible_faces=faces,
        fit_rms_m=fit_rms,
        n_points=len(pts),
    )
