"""Running FoundationPose from depth alone, with no colour camera.

FoundationPose does not have a depth-only mode. Both of its networks take a
six-channel input built as ``cat([rgb_crop, xyz_crop], dim=1)`` -- once for the
rendered hypothesis (branch A) and once for the observation (branch B) --
in ``learning/training/predict_pose_refine.py`` and ``predict_score.py`` alike.
``register(K, rgb, depth, ob_mask)`` and ``track_one(rgb, depth, K)`` both take
``rgb`` positionally and warp it before anything else runs, so passing ``None``
is a crash, not a degraded mode.

Depth-only therefore means *substituting* the colour channel, not omitting it.
The substitution here is not arbitrary. Branch A renders an untextured mesh with
``use_light=True``, which in ``Utils.nvdiffrast_render`` evaluates to

    I = clip(albedo * (w_ambient + w_diffuse * clip(-n_z, 0, 1)), 0, 1)

with ``w_ambient=0.8``, ``w_diffuse=0.5``, a headlight along the optical axis,
and background pixels forced to black. That is a shading map and nothing else --
no texture, no albedo variation. So the closest achievable match for branch B is
the same formula evaluated on normals estimated from the depth image, which is
what :func:`depth_to_pseudo_rgb` computes. Both branches then live in the same
domain, which is the property the render-and-compare architecture depends on.

What this does and does not buy you
-----------------------------------
Translation survives it well. FoundationPose's translation is seeded by
``guess_translation``, which is pure depth and mask, and refined against the XYZ
half of the input; the RGB half mostly disambiguates *rotation*. Since distance
is a translation question, and since a cuboid's rotation is only recoverable up
to its own symmetry anyway, depth-only costs little for this particular task.

Rotation and the pose *score* degrade. The scorer was trained on real imagery
and a shading map is out of distribution for it, so treat ``score`` as ordinal
within a frame, not as a calibrated confidence. The confidence reported on
:class:`PoseResult` comes from the geometric agreement between the returned pose
and the observed points instead -- see :func:`boxrange.ranging.estimate_range`.

And for a box on a floor, this is the expensive way to get the answer. The
geometric pipeline in :mod:`boxrange.pipeline` needs no GPU, no mesh, and no
network, and measures the same distance more accurately. Reach for this module
when you want FoundationPose's actual strengths -- full 6D pose of a specific
known object, heavy occlusion, or telling apart objects a plane fit cannot --
and want them without a colour stream.

Requirements: FoundationPose itself, CUDA, nvdiffrast, torch, and trimesh. None
of them import at module load, so the helpers below stay testable anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .frames import Frame
from .geometry import OrientedBox, Plane, deproject, fit_box_on_plane
from .intrinsics import CameraIntrinsics
from .pipeline import BoxDetection
from .ranging import RangeEstimate, estimate_range
from .segment import Candidate, PlaneClusterDetector, depth_discontinuities

# Lighting constants copied from FoundationPose's Utils.nvdiffrast_render
# defaults. They are the render-branch appearance model, so they belong to
# FoundationPose, not to a preference here -- change them only to track upstream.
W_AMBIENT = 0.8
W_DIFFUSE = 0.5

# make_mesh_tensors assigns [128, 128, 128] to a mesh with no vertex colours.
# box_mesh sets the same value explicitly rather than relying on trimesh's own
# default, which is a different grey and would put the two branches at
# different brightness for no reason.
DEFAULT_ALBEDO = 128.0 / 255.0


# --------------------------------------------------------------------------
# Depth -> the colour channel FoundationPose insists on
# --------------------------------------------------------------------------


def surface_normals(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    smooth_px: int = 9,
    step: int = 2,
    jump_sigmas: float = 6.0,
    jump_floor_m: float = 0.012,
) -> np.ndarray:
    """Per-pixel surface normals from a depth image, (H, W, 3) float32.

    Normals point *towards* the camera -- ``n . P <= 0`` for the point ``P`` they
    sit on -- and a face viewed head-on gives ``n = (0, 0, -1)``. That is the
    convention FoundationPose's shading term expects. Pixels where the normal is
    undefined come back NaN.

    Note the invariant is ``n . P <= 0`` and *not* ``n_z <= 0``. Across a 90 deg
    field of view the edge rays are 45 deg off axis, so a surface seen at grazing
    incidence out there can face the camera and still have a positive z
    component. Asserting the simpler-looking condition would flag correct
    normals as broken.

    Smoothing is not optional. A raw central difference on stereo depth is
    almost pure noise: at 2 m a Gemini 335 carries ~30 mm of range sigma across
    a ~3 mm pixel footprint, so the sample-to-sample slope is meaningless. The
    ``smooth_px`` box filter over the point cloud and the ``step``-pixel
    difference stencil both trade angular resolution for a usable normal, and
    the defaults are sized for a box face at a couple of metres, not for
    resolving fine geometry.

    Smoothing across a depth discontinuity would instead invent a smooth ramp
    joining a box edge to the floor behind it, so discontinuous pixels are
    dropped *before* the filter runs rather than after.

    Measured against analytic normals on the synthetic rig, Gemini 335 noise
    model at 640x400, median angular error over the box's pixels:

    ===========  ======  ======  ======  ======
    smooth/step    1.0 m   1.6 m   2.5 m   3.5 m
    ===========  ======  ======  ======  ======
    1 / 1 (raw)   ~52 deg   --      --      --
    5 / 1         8.0     11.8    18.3    29.1
    9 / 2         3.1      5.9    10.9    19.8
    13 / 3        2.1      7.9    15.4    29.5
    ===========  ======  ======  ======  ======

    So the default is 9 / 2, and the shading map carries real signal only inside
    roughly the Gemini 335's quoted optimal band of 0.26-3 m. Past that the
    normals wash out and the substituted colour channel approaches a flat grey,
    which is the point at which FoundationPose is running on depth alone in
    substance as well as in name.
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    cloud = deproject(depth_m, intrinsics).astype(np.float32)

    tol = jump_sigmas * intrinsics.range_sigma(np.where(depth_m > 0, depth_m, 1.0))
    edge = depth_discontinuities(depth_m, (tol + jump_floor_m).astype(np.float32))
    # A pixel next to a jump still has a neighbour across it inside the stencil.
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    cloud[edge] = np.nan

    if smooth_px > 1:
        cloud = _nan_box_filter(cloud, smooth_px)

    # Central differences along the image axes. Both are NaN-propagating, which
    # is what should happen: a normal needs all four neighbours.
    du = np.full_like(cloud, np.nan)
    dv = np.full_like(cloud, np.nan)
    du[:, step:-step] = cloud[:, 2 * step :] - cloud[:, : -2 * step]
    dv[step:-step, :] = cloud[2 * step :, :] - cloud[: -2 * step, :]

    # cross(du, dv) points away from the camera for a front-facing surface
    # (+u right, +v down, +z forward), so negate to get the viewing-side normal.
    n = -np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.where(norm > 1e-12, n / norm, np.nan)

    # A surface the camera can see faces the camera, so enforce it. On a patch
    # seen almost edge-on the smoothed differences are nearly parallel and depth
    # noise flips the cross product outright -- ~0.02% of pixels on a synthetic
    # frame at 1.6 m. Shading is even in the normal near grazing incidence so
    # the visual cost is nil, but leaving them inverted would make the docstring
    # invariant a lie and quietly break anyone reusing these normals.
    #
    # Test against the pixel's viewing ray rather than its 3D position. The two
    # have the same sign because P = z * ray with z > 0, but the ray comes from
    # the intrinsics alone and carries no depth noise, whereas ``cloud`` here has
    # been smoothed -- and on exactly the grazing patches at issue, that
    # smoothing is enough to move the position across the sign boundary.
    ray = np.stack(
        (
            (np.arange(cloud.shape[1], dtype=np.float32) - intrinsics.cx) / intrinsics.fx
            + np.zeros((cloud.shape[0], 1), dtype=np.float32),
            (np.arange(cloud.shape[0], dtype=np.float32) - intrinsics.cy)[:, None]
            / intrinsics.fy
            + np.zeros((1, cloud.shape[1]), dtype=np.float32),
            np.ones(cloud.shape[:2], dtype=np.float32),
        ),
        axis=-1,
    )
    with np.errstate(invalid="ignore"):
        flip = np.nansum(n * ray, axis=-1) > 0
    n[flip] *= -1.0
    return n.astype(np.float32)


def _nan_box_filter(cloud: np.ndarray, ksize: int) -> np.ndarray:
    """Box filter that ignores NaNs instead of spreading them.

    ``cv2.blur`` propagates a single NaN across the whole kernel footprint, which
    on a frame with 2% dropout erases most of the image. Summing values and
    weights separately and dividing gives the mean of whatever was valid.
    """
    valid = np.isfinite(cloud).all(axis=-1)
    filled = np.where(valid[..., None], cloud, 0.0).astype(np.float32)

    k = (ksize, ksize)
    num = cv2.blur(filled, k, borderType=cv2.BORDER_REPLICATE)
    den = cv2.blur(valid.astype(np.float32), k, borderType=cv2.BORDER_REPLICATE)

    out = np.full_like(cloud, np.nan)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok, None]
    return out


def shade(
    normals: np.ndarray,
    *,
    albedo: float = DEFAULT_ALBEDO,
    w_ambient: float = W_AMBIENT,
    w_diffuse: float = W_DIFFUSE,
) -> np.ndarray:
    """FoundationPose's render-branch lighting term, evaluated on ``normals``.

    Mirrors ``Utils.nvdiffrast_render`` under ``use_light=True``: a headlight
    along ``+z``, so the diffuse term is ``clip(-n_z, 0, 1)``, combined as
    ``albedo * (w_ambient + w_diffuse * diffuse)`` and clipped to [0, 1].
    Undefined normals render as background, which that function sets to black.

    Returns (H, W) float32 in [0, 1].
    """
    nz = normals[..., 2]
    with np.errstate(invalid="ignore"):
        diffuse = np.clip(-nz, 0.0, 1.0)
    intensity = albedo * (w_ambient + w_diffuse * diffuse)
    return np.clip(np.nan_to_num(intensity, nan=0.0), 0.0, 1.0).astype(np.float32)


def depth_to_pseudo_rgb(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    albedo: float = DEFAULT_ALBEDO,
    **normal_kw,
) -> np.ndarray:
    """The (H, W, 3) uint8 image to hand FoundationPose in place of colour.

    Grey replicated across three channels, so RGB-versus-BGR order does not
    arise -- worth stating because FoundationPose does expect RGB and a silent
    channel swap is the usual bug at this seam.
    """
    normals = surface_normals(depth_m, intrinsics, **normal_kw)
    grey = shade(normals, albedo=albedo)
    return np.repeat((grey * 255.0).round().astype(np.uint8)[..., None], 3, axis=2)


# --------------------------------------------------------------------------
# The mesh FoundationPose needs, which for a box is three numbers
# --------------------------------------------------------------------------


def box_mesh_arrays(extents_m) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (8, 3) and triangles (12, 3) for a cuboid centred on the origin.

    Centred is load-bearing. ``FoundationPose.reset_object`` subtracts the mesh's
    bounding-box centre internally and ``register`` returns the pose composed
    back onto the *original* frame, so an origin-centred mesh makes the returned
    translation the box centre exactly, with no offset to undo in
    :func:`pose_to_box`.

    Winding is counter-clockwise seen from outside, which is what makes the
    vertex normals point outwards and therefore what makes the shading term
    agree with :func:`surface_normals`.
    """
    e = np.asarray(extents_m, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(e)) or np.any(e <= 0):
        raise ValueError(f"extents must be finite and positive, got {extents_m}")

    h = e / 2.0
    signs = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    vertices = signs * h
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # -z
            [4, 5, 6], [4, 6, 7],  # +z
            [0, 1, 5], [0, 5, 4],  # -y
            [3, 7, 6], [3, 6, 2],  # +y
            [0, 4, 7], [0, 7, 3],  # -x
            [1, 2, 6], [1, 6, 5],  # +x
        ],
        dtype=np.int64,
    )
    return vertices, faces


def box_mesh(extents_m, *, albedo_255: int = 128):
    """A ``trimesh.Trimesh`` cuboid, ready to hand to ``FoundationPose``.

    The model-based path wants a CAD model; for a cuboid that model is its three
    side lengths, so nothing has to be authored or loaded. Vertex colours are set
    explicitly to the same grey ``make_mesh_tensors`` falls back to, so the
    rendered branch and :func:`depth_to_pseudo_rgb` share one albedo.
    """
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "trimesh is required to build the FoundationPose mesh. It is already "
            "a FoundationPose dependency, so `pip install trimesh` inside the "
            "same environment."
        ) from exc

    vertices, faces = box_mesh_arrays(extents_m)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = np.tile(
        np.array([albedo_255, albedo_255, albedo_255, 255], dtype=np.uint8), (len(vertices), 1)
    )
    return mesh


def pose_to_box(pose: np.ndarray, extents_m) -> OrientedBox:
    """A 4x4 object-in-camera pose plus known extents -> :class:`OrientedBox`.

    Valid only for the origin-centred mesh :func:`box_mesh` builds; see the note
    there about why the translation needs no correction.
    """
    T = np.asarray(pose, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"pose must be 4x4, got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError("pose contains non-finite values")
    return OrientedBox(
        center=T[:3, 3].copy(),
        R=T[:3, :3].copy(),
        extents=np.asarray(extents_m, dtype=np.float64).reshape(3).copy(),
    )


def box_to_pose(box: OrientedBox) -> np.ndarray:
    """Inverse of :func:`pose_to_box`: an oriented box as a 4x4 pose."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = box.R
    T[:3, 3] = box.center
    return T


# --------------------------------------------------------------------------
# The mask FoundationPose needs, which normally comes from an RGB network
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Seed:
    """What the depth front end hands FoundationPose to start from."""

    mask: np.ndarray  # (H, W) bool, the object's pixels
    candidate: Candidate
    plane: Plane
    box: OrientedBox | None  # geometric fit, the fallback source of extents


def seed_from_depth(
    frame: Frame, detector: PlaneClusterDetector | None = None
) -> Seed | None:
    """Segment the object from depth, replacing the usual RGB mask source.

    FoundationPose's demos get their first-frame mask from SAM or Mask R-CNN,
    both of which need the colour image this module does not have. For an object
    standing on a support plane the depth segmentation already in this package
    does the same job without a network: fit the floor, drop it, take the largest
    cluster that remains.

    The geometric box fit rides along because it is the natural source of the
    extents the mesh needs when the caller has not measured the box.
    """
    detector = detector or PlaneClusterDetector()
    candidates, plane = detector.detect(frame.depth_m, frame.intrinsics, frame.color)
    if plane is None or not candidates:
        return None
    best = candidates[0]  # detect() returns them largest-first
    return Seed(
        mask=best.mask,
        candidate=best,
        plane=plane,
        box=fit_box_on_plane(best.points, plane),
    )


# --------------------------------------------------------------------------
# The estimator wrapper
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseResult:
    """One FoundationPose result, converted into this package's units."""

    pose: np.ndarray  # (4, 4) object-in-camera
    box: OrientedBox
    range: RangeEstimate
    plane: Plane
    candidate: Candidate  # the points and mask the range was measured from
    tracked: bool  # False on the registration frame, True while tracking
    score: float = float("nan")  # FoundationPose's own; see the module docstring

    @property
    def distance_m(self) -> float:
        """Camera to the nearest point on the box, metres."""
        return self.range.surface_m

    @property
    def mask(self) -> np.ndarray:
        return self.candidate.mask

    def to_detection(self, track_id: int = 0) -> BoxDetection:
        """As a :class:`~boxrange.pipeline.BoxDetection`.

        Both engines then feed one output path -- the JSON writer, the overlay,
        and anything downstream -- so a FoundationPose distance and a geometric
        one stay directly comparable instead of arriving in two shapes that have
        to be reconciled by eye.
        """
        return BoxDetection(
            box=self.box,
            range=self.range,
            candidate=self.candidate,
            plane=self.plane,
            track_id=track_id,
            smoothed_surface_m=self.range.surface_m,
        )


def require_foundationpose():
    """Import FoundationPose and its render backend, or explain what is missing.

    Kept out of module scope because the import chain pulls in torch, nvdiffrast
    and a CUDA context; deferring it is what lets every helper above -- and the
    tests for them -- run on a laptop with none of that installed.

    Exposed rather than private so a caller can fail fast. The estimator is
    otherwise built lazily on the first frame, which on a robot means
    discovering the environment is incomplete only after the camera is streaming.

    Returns ``(nvdiffrast.torch, FoundationPose, ScorePredictor,
    PoseRefinePredictor)``.
    """
    try:
        import nvdiffrast.torch as dr
        from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    except ImportError as exc:  # pragma: no cover - needs the CUDA stack
        raise ImportError(
            "FoundationPose was not importable. Clone NVlabs/FoundationPose, "
            "install its environment (CUDA, torch, nvdiffrast, trimesh), "
            "download the pretrained weights, and run with its repo root on "
            "PYTHONPATH. There is no CPU fallback -- the refiner and scorer both "
            "call .cuda() unconditionally. Pass estimator=... to inject an "
            "already-constructed one, or use GeometricEstimator to exercise "
            "everything around the network without a GPU."
        ) from exc
    return dr, FoundationPose, ScorePredictor, PoseRefinePredictor


class DepthOnlyFoundationPose:
    """FoundationPose driven by depth alone.

    Typical use, on a machine with the CUDA stack installed::

        est = DepthOnlyFoundationPose()          # extents learned from frame 1
        with OrbbecSource() as source:
            for frame in source:
                result = est.process(frame)
                if result is not None:
                    print(result.distance_m, result.range.sigma_m)

    ``extents_m`` is the box's side lengths in metres. Leave it None to take them
    from the depth fit on the registration frame, which is the honest default
    when the box is not known in advance -- but a measured box is strictly
    better, because a wrong mesh biases every pose that follows and nothing
    downstream can detect that.

    ``estimator`` accepts a pre-built ``FoundationPose`` instance. Passing one is
    how the tests exercise this class without CUDA, and how a caller reuses a
    single loaded set of network weights across several objects.
    """

    def __init__(
        self,
        *,
        extents_m=None,
        detector: PlaneClusterDetector | None = None,
        estimator=None,
        register_iterations: int = 5,
        track_iterations: int = 2,
        albedo: float = DEFAULT_ALBEDO,
        normal_kwargs: dict | None = None,
        redetect_after: int = 30,
    ) -> None:
        self.extents_m = (
            None if extents_m is None else np.asarray(extents_m, dtype=np.float64).reshape(3)
        )
        self.detector = detector or PlaneClusterDetector()
        self.register_iterations = register_iterations
        self.track_iterations = track_iterations
        self.albedo = albedo
        self.normal_kwargs = normal_kwargs or {}
        self.redetect_after = redetect_after

        self._estimator = estimator
        self._registered = False
        self._plane: Plane | None = None
        self._frames_since_register = 0

    # -- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        """Forget the tracked pose so the next frame re-registers."""
        self._registered = False
        self._frames_since_register = 0

    def _build_estimator(self):
        """Construct the real FoundationPose. Needs ``extents_m`` to be known."""
        dr, FoundationPose, ScorePredictor, PoseRefinePredictor = require_foundationpose()
        mesh = box_mesh(self.extents_m)
        return FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(),
        )

    # -- per-frame --------------------------------------------------------

    def process(self, frame: Frame) -> PoseResult | None:
        """Register on the first frame, then track; re-register when stale.

        Re-registration is periodic rather than on-failure because there is no
        reliable failure signal available here: the pose score is unreliable on
        pseudo-colour (see the module docstring), and a drifted tracker returns
        a confident-looking pose indefinitely. ``redetect_after`` bounds how long
        drift can accumulate. Set it to 0 to track forever.
        """
        if self._registered and self.redetect_after and (
            self._frames_since_register >= self.redetect_after
        ):
            self._registered = False

        if not self._registered:
            return self.register(frame)
        return self.track(frame)

    def register(self, frame: Frame) -> PoseResult | None:
        """Full global registration on ``frame``. Returns None if nothing found."""
        seed = seed_from_depth(frame, self.detector)
        if seed is None:
            return None

        if self.extents_m is None:
            if seed.box is None:
                return None
            # One-sided bias warning: the depth fit reads slightly small in
            # depth-along-view because only one or two faces are ever observed.
            # Measuring the box with a tape and passing extents_m avoids baking
            # that into the mesh for the whole run.
            self.extents_m = np.asarray(seed.box.extents, dtype=np.float64).copy()

        if self._estimator is None:
            self._estimator = self._build_estimator()

        rgb = self.pseudo_rgb(frame)
        pose = self._estimator.register(
            K=frame.intrinsics.matrix,
            rgb=rgb,
            depth=frame.depth_m,
            ob_mask=seed.mask.astype(np.uint8),
            iteration=self.register_iterations,
        )

        self._registered = True
        self._frames_since_register = 0
        self._plane = seed.plane
        return self._result(pose, frame, seed.mask, seed.plane, tracked=False)

    def track(self, frame: Frame) -> PoseResult | None:
        """Refine the previous pose against ``frame``. Cheap; no segmentation."""
        if not self._registered or self._estimator is None:
            return self.register(frame)

        pose = self._estimator.track_one(
            rgb=self.pseudo_rgb(frame),
            depth=frame.depth_m,
            K=frame.intrinsics.matrix,
            iteration=self.track_iterations,
        )
        self._frames_since_register += 1

        # Tracking skips segmentation, so the mask for ranging is the pose's own
        # projected footprint rather than a fresh detection.
        box = pose_to_box(pose, self.extents_m)
        mask = self.project_mask(box, frame)
        assert self._plane is not None
        return self._result(pose, frame, mask, self._plane, tracked=True)

    # -- helpers ----------------------------------------------------------

    def pseudo_rgb(self, frame: Frame) -> np.ndarray:
        """The shading image standing in for the colour stream on this frame."""
        return depth_to_pseudo_rgb(
            frame.depth_m, frame.intrinsics, albedo=self.albedo, **self.normal_kwargs
        )

    @staticmethod
    def project_mask(box: OrientedBox, frame: Frame, *, pad_px: int = 0) -> np.ndarray:
        """Pixels inside the box's projected silhouette.

        The convex hull of the eight projected corners is the exact silhouette of
        a convex solid, so no rasteriser is needed. Corners behind the camera are
        dropped rather than projected, which would otherwise fold them to the
        wrong side of the image.
        """
        intr = frame.intrinsics
        corners = box.corners()
        front = corners[corners[:, 2] > 1e-6]
        mask = np.zeros(frame.depth_m.shape, dtype=bool)
        if len(front) < 3:
            return mask

        u = intr.fx * front[:, 0] / front[:, 2] + intr.cx
        v = intr.fy * front[:, 1] / front[:, 2] + intr.cy
        pts = np.stack((u, v), axis=1).astype(np.float32)

        hull = cv2.convexHull(pts)
        filled = np.zeros(frame.depth_m.shape, dtype=np.uint8)
        cv2.fillConvexPoly(filled, hull.astype(np.int32), 1)
        if pad_px > 0:
            k = np.ones((2 * pad_px + 1, 2 * pad_px + 1), np.uint8)
            filled = cv2.dilate(filled, k)
        return filled.astype(bool)

    def _result(
        self,
        pose: np.ndarray,
        frame: Frame,
        mask: np.ndarray,
        plane: Plane,
        *,
        tracked: bool,
    ) -> PoseResult | None:
        """Convert a FoundationPose pose into the package's range report.

        Ranging reuses :func:`boxrange.ranging.estimate_range` on the points the
        mask selects, so a FoundationPose distance and a geometric one are the
        same quantity under the same definition and can be compared directly.
        Its ``fit_rms_m`` becomes a real check here rather than a formality: it
        measures how far the observed points sit from the faces of the pose
        FoundationPose returned, which is the one signal available for whether
        that pose is right.
        """
        box = pose_to_box(pose, self.extents_m)
        cloud = deproject(frame.depth_m, frame.intrinsics)
        points = cloud[mask]
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return None

        score = float("nan")
        scores = getattr(self._estimator, "scores", None)
        if scores is not None and len(scores) > 0:
            score = float(np.asarray(scores[0].cpu() if hasattr(scores[0], "cpu") else scores[0]))

        candidate = Candidate(points=points, mask=mask, label=1)
        return PoseResult(
            pose=np.asarray(pose, dtype=np.float64),
            box=box,
            range=estimate_range(points, box, frame.intrinsics, plane=plane),
            plane=plane,
            candidate=candidate,
            tracked=tracked,
            score=score,
        )


# --------------------------------------------------------------------------
# A stand-in, for testing the plumbing without CUDA
# --------------------------------------------------------------------------


@dataclass
class GeometricEstimator:
    """Drop-in for ``FoundationPose`` that solves the pose geometrically.

    Same ``register`` / ``track_one`` signatures, so it exercises every line of
    :class:`DepthOnlyFoundationPose` -- the mask, the mesh, the pseudo-colour,
    the pose-to-range conversion -- on a machine with no GPU. It ignores ``rgb``,
    which is exactly the point: it isolates whether the plumbing is right from
    whether the network likes the substituted colour channel.

    It is not an approximation of FoundationPose and makes no claim about what
    FoundationPose would return. Do not use it to characterise accuracy.
    """

    detector: PlaneClusterDetector = field(default_factory=PlaneClusterDetector)
    scores: list = field(default_factory=list)

    def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5):
        intr = CameraIntrinsics.from_matrix(np.asarray(K), depth.shape[1], depth.shape[0])
        cloud = deproject(depth, intr)
        mask = np.asarray(ob_mask).astype(bool)
        points = cloud[mask]
        points = points[np.isfinite(points).all(axis=1)]

        from .geometry import fit_plane_ransac

        finite = cloud[np.isfinite(cloud).all(axis=-1)]
        plane = fit_plane_ransac(finite, up_hint=np.array([0.0, -1.0, 0.0]))
        if plane is None:
            raise RuntimeError("no support plane")
        box = fit_box_on_plane(points, plane)
        if box is None:
            raise RuntimeError("no box fit")

        self._last = box_to_pose(box)
        self.scores = [1.0]
        return self._last

    def track_one(self, rgb, depth, K, iteration=2, extra=None):
        return self._last
