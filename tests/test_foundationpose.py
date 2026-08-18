"""Depth-only FoundationPose: the mesh, the mask, the substituted colour channel.

The estimator itself needs CUDA, so these tests inject
:class:`GeometricEstimator` in its place. That deliberately splits the two
questions -- is the plumbing right, and does the network like a shading map --
and only the first is answerable without a GPU. Nothing here claims anything
about FoundationPose's accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxrange.foundationpose import (
    DEFAULT_ALBEDO,
    W_AMBIENT,
    W_DIFFUSE,
    DepthOnlyFoundationPose,
    GeometricEstimator,
    _nan_box_filter,
    box_mesh_arrays,
    box_to_pose,
    depth_to_pseudo_rgb,
    pose_to_box,
    seed_from_depth,
    shade,
    surface_normals,
)
from boxrange.frames import Frame, SyntheticScene, render_depth
from boxrange.geometry import deproject
from boxrange.intrinsics import CameraIntrinsics, gemini335_depth
from boxrange.ranging import closest_point_on_box

X2_INTR = gemini335_depth(640, 400)


def make_frame(scene: SyntheticScene, *, noise: bool = True, seed: int = 0, intr=X2_INTR):
    return Frame(
        depth_m=render_depth(scene, intr, noise=noise, rng=np.random.default_rng(seed)),
        intrinsics=intr,
    )


def analytic_normals(scene: SyntheticScene, intr: CameraIntrinsics):
    """Ground-truth normals by ray-casting: the face normal of whatever was hit."""
    cloud = deproject(render_depth(scene, intr, noise=False, dropout=0.0), intr)
    box, plane = scene.truth_box(), scene.ground_plane()
    flat = cloud.reshape(-1, 3)

    local = (flat - box.center) @ box.R
    half = box.extents / 2.0
    on_box = (np.abs(local) <= half + 2e-3).all(axis=1)
    axis = np.argmin(np.abs(np.abs(local) - half), axis=1)
    sign = np.sign(local[np.arange(len(local)), axis])

    n = (box.R[:, axis] * sign).T
    n = np.where(on_box[:, None], n, plane.normal[None, :])
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    n[(n * flat).sum(1) > 0] *= -1  # face the camera
    return n.reshape(cloud.shape), on_box.reshape(cloud.shape[:2])


# --------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------


def test_box_mesh_is_centred_with_the_requested_extents():
    v, f = box_mesh_arrays([0.40, 0.30, 0.25])
    assert v.shape == (8, 3) and f.shape == (12, 3)
    assert np.allclose(v.mean(axis=0), 0.0)
    assert np.allclose(v.max(axis=0) - v.min(axis=0), [0.40, 0.30, 0.25])


def test_box_mesh_triangles_wind_outward():
    """Outward winding is what makes vertex normals -- and so shading -- correct.

    Reversed winding renders the box inside-out and puts the substituted colour
    channel exactly out of phase with the observation, which would be invisible
    in every other test here.
    """
    v, f = box_mesh_arrays([0.4, 0.3, 0.25])
    tri = v[f]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    centroids = tri.mean(axis=1)
    assert np.all((normals * centroids).sum(axis=1) > 0)


def test_box_mesh_rejects_degenerate_extents():
    for bad in ([0.4, 0.0, 0.25], [0.4, -0.3, 0.25], [0.4, np.nan, 0.25]):
        with pytest.raises(ValueError):
            box_mesh_arrays(bad)


def test_pose_and_box_round_trip():
    scene = SyntheticScene(forward_m=1.8, yaw=np.deg2rad(31.0))
    box = scene.truth_box()
    back = pose_to_box(box_to_pose(box), box.extents)
    assert np.allclose(back.center, box.center)
    assert np.allclose(back.R, box.R)
    assert np.allclose(back.extents, box.extents)


def test_pose_to_box_rejects_malformed_poses():
    with pytest.raises(ValueError):
        pose_to_box(np.eye(3), [1, 1, 1])
    bad = np.eye(4)
    bad[0, 3] = np.nan
    with pytest.raises(ValueError):
        pose_to_box(bad, [1, 1, 1])


# --------------------------------------------------------------------------
# The substituted colour channel
# --------------------------------------------------------------------------


def test_shade_matches_foundationpose_lighting_exactly():
    """The whole point of the substitution is domain match, so check the formula.

    Values come from ``Utils.nvdiffrast_render`` with ``use_light=True``:
    ``albedo * (0.8 + 0.5 * clip(-n_z, 0, 1))``, clipped, background black.
    """
    head_on = np.array([[[0.0, 0.0, -1.0]]])
    grazing = np.array([[[1.0, 0.0, 0.0]]])
    away = np.array([[[0.0, 0.0, 1.0]]])
    undefined = np.full((1, 1, 3), np.nan)

    assert shade(head_on)[0, 0] == pytest.approx(DEFAULT_ALBEDO * (W_AMBIENT + W_DIFFUSE))
    assert shade(grazing)[0, 0] == pytest.approx(DEFAULT_ALBEDO * W_AMBIENT)
    assert shade(away)[0, 0] == pytest.approx(DEFAULT_ALBEDO * W_AMBIENT)
    assert shade(undefined)[0, 0] == 0.0


def test_shade_stays_in_gamut():
    rng = np.random.default_rng(0)
    n = rng.normal(size=(64, 64, 3))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    out = shade(n, albedo=1.0)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_pseudo_rgb_is_a_grey_uint8_image_with_black_background():
    frame = make_frame(SyntheticScene(forward_m=1.6))
    img = depth_to_pseudo_rgb(frame.depth_m, frame.intrinsics)

    assert img.dtype == np.uint8
    assert img.shape == (*frame.depth_m.shape, 3)
    # Grey replicated, so no RGB/BGR ordering bug is possible at this seam.
    assert np.array_equal(img[..., 0], img[..., 1])
    assert np.array_equal(img[..., 1], img[..., 2])
    # Bounded by the lighting model rather than by 255.
    assert img.max() <= round(255 * DEFAULT_ALBEDO * (W_AMBIENT + W_DIFFUSE))
    assert img.min() == 0  # dropout and edges render as background


def test_pseudo_rgb_separates_the_box_from_the_floor():
    """A flat grey image would satisfy every check above and be useless.

    The property that matters is that the shading map *distinguishes surfaces*:
    the box's front face and the floor it stands on have different normals, so
    they must land at different brightnesses. Testing the image's overall
    standard deviation instead would mostly measure how much of the frame is
    floor, which is a fact about the scene rather than about the shading.
    """
    scene = SyntheticScene(forward_m=1.4)
    frame = make_frame(scene, noise=False)
    img = depth_to_pseudo_rgb(frame.depth_m, frame.intrinsics)[..., 0].astype(float)

    seed = seed_from_depth(frame)
    assert seed is not None
    on_box = seed.mask & (img > 0)
    on_floor = (~seed.mask) & (img > 0)

    gap = abs(img[on_box].mean() - img[on_floor].mean())
    # The whole lighting model spans albedo*0.8 to albedo*1.3, i.e. 64 grey
    # levels, so measure the gap against that range rather than against 255.
    span = 255 * DEFAULT_ALBEDO * W_DIFFUSE
    assert gap > 0.15 * span, f"box and floor differ by only {gap:.1f} of {span:.0f} levels"


# --------------------------------------------------------------------------
# Normals
# --------------------------------------------------------------------------


def test_normals_face_the_camera():
    """Every visible surface's normal must point back at the camera.

    The invariant is ``n . P <= 0``, not ``n_z <= 0``: across the Gemini 335's
    90 degree field of view the edge rays run 45 degrees off axis, so a grazing
    surface out there faces the camera while still having a positive z. Depth
    noise also inverts the cross product on near-edge-on patches, which
    :func:`surface_normals` corrects rather than passes through -- this is the
    test for that correction.
    """
    frame = make_frame(SyntheticScene(forward_m=1.6))
    n = surface_normals(frame.depth_m, frame.intrinsics)
    cloud = deproject(frame.depth_m, frame.intrinsics)

    ok = np.isfinite(n).all(axis=-1) & np.isfinite(cloud).all(axis=-1)
    assert ok.mean() > 0.5
    assert np.all((n[ok] * cloud[ok]).sum(axis=-1) <= 1e-6)
    # And head-on surfaces still come out with a strongly negative z.
    assert n[ok][:, 2].min() < -0.9


def test_normals_of_a_fronto_parallel_wall_point_straight_back():
    intr = CameraIntrinsics(width=200, height=200, fx=200.0, fy=200.0, cx=99.5, cy=99.5)
    depth = np.full((200, 200), 2.0, dtype=np.float32)
    n = surface_normals(depth, intr, smooth_px=5, step=2)
    inner = n[20:-20, 20:-20]
    assert np.allclose(inner, np.array([0.0, 0.0, -1.0]), atol=1e-3)


def test_normals_beat_raw_differencing_on_noisy_depth():
    """Smoothing is the difference between a usable normal and pure noise."""
    scene = SyntheticScene(forward_m=1.6)
    truth, on_box = analytic_normals(scene, X2_INTR)
    frame = make_frame(scene, seed=1)

    def median_error(**kw):
        n = surface_normals(frame.depth_m, frame.intrinsics, **kw)
        ok = np.isfinite(n).all(axis=-1) & on_box
        dot = np.clip((n[ok] * truth[ok]).sum(axis=-1), -1.0, 1.0)
        return float(np.median(np.rad2deg(np.arccos(dot))))

    raw = median_error(smooth_px=1, step=1)
    default = median_error()
    assert raw > 30.0, "expected raw differencing to be dominated by depth noise"
    assert default < 10.0
    assert default < raw / 3.0


def test_normals_reject_a_bad_step():
    with pytest.raises(ValueError):
        surface_normals(np.ones((8, 8), np.float32), X2_INTR, step=0)


def test_nan_box_filter_does_not_spread_holes():
    cloud = np.ones((16, 16, 3), dtype=np.float32)
    cloud[8, 8] = np.nan
    out = _nan_box_filter(cloud, 5)
    # cv2.blur would have wiped the whole 5x5 footprint.
    assert np.isfinite(out).all()
    assert np.allclose(out, 1.0)


def test_nan_box_filter_keeps_fully_empty_regions_empty():
    cloud = np.full((16, 16, 3), np.nan, dtype=np.float32)
    assert not np.isfinite(_nan_box_filter(cloud, 5)).any()


# --------------------------------------------------------------------------
# The mask, standing in for SAM
# --------------------------------------------------------------------------


def test_seed_from_depth_finds_the_box_without_colour():
    scene = SyntheticScene(forward_m=1.6)
    seed = seed_from_depth(make_frame(scene))
    assert seed is not None
    assert seed.box is not None
    assert seed.mask.any()
    # The mask should land on the box, not on the floor behind it.
    truth = scene.truth_box()
    assert np.linalg.norm(seed.box.center - truth.center) < 0.10


def test_seed_from_depth_returns_none_on_an_empty_frame():
    frame = Frame(depth_m=np.zeros((120, 160), np.float32), intrinsics=X2_INTR)
    assert seed_from_depth(frame) is None


def test_projected_mask_covers_the_box_and_nothing_behind_the_camera():
    scene = SyntheticScene(forward_m=1.6)
    frame = make_frame(scene)
    seed = seed_from_depth(frame)
    assert seed is not None

    projected = DepthOnlyFoundationPose.project_mask(scene.truth_box(), frame)
    assert projected.any()
    # The silhouette of the true box must contain most of the segmented pixels.
    assert (seed.mask & projected).sum() > 0.8 * seed.mask.sum()

    behind = scene.truth_box()
    behind = type(behind)(behind.center - [0, 0, 10.0], behind.R, behind.extents)
    assert not DepthOnlyFoundationPose.project_mask(behind, frame).any()


# --------------------------------------------------------------------------
# End to end, with the estimator stubbed out
# --------------------------------------------------------------------------


def test_end_to_end_reports_the_distance_to_the_nearest_face():
    scene = SyntheticScene(forward_m=1.6)
    est = DepthOnlyFoundationPose(estimator=GeometricEstimator())
    result = est.process(make_frame(scene))

    assert result is not None and not result.tracked
    truth = float(np.linalg.norm(closest_point_on_box(scene.truth_box())))
    assert result.distance_m == pytest.approx(truth, abs=0.06)
    assert result.range.sigma_m > 0.0
    assert 0.0 <= result.range.confidence <= 1.0
    assert result.pose.shape == (4, 4)


def test_extents_are_bootstrapped_from_depth_when_not_supplied():
    scene = SyntheticScene(forward_m=1.6, box_size=(0.40, 0.30, 0.25))
    est = DepthOnlyFoundationPose(estimator=GeometricEstimator())
    assert est.extents_m is None
    est.process(make_frame(scene))
    assert est.extents_m is not None
    assert np.allclose(est.extents_m, [0.40, 0.30, 0.25], atol=0.05)


def test_supplied_extents_are_used_verbatim():
    """A measured box must override the fit, since the fit is the weaker input."""
    scene = SyntheticScene(forward_m=1.6)
    est = DepthOnlyFoundationPose(extents_m=[0.40, 0.30, 0.25], estimator=GeometricEstimator())
    result = est.process(make_frame(scene))
    assert result is not None
    assert np.allclose(result.box.extents, [0.40, 0.30, 0.25])


def test_first_frame_registers_and_later_frames_track():
    scene = SyntheticScene(forward_m=1.6)
    est = DepthOnlyFoundationPose(estimator=GeometricEstimator(), redetect_after=0)
    flags = [est.process(make_frame(scene, seed=i)).tracked for i in range(4)]
    assert flags == [False, True, True, True]


def test_redetect_after_forces_a_fresh_registration():
    scene = SyntheticScene(forward_m=1.6)
    est = DepthOnlyFoundationPose(estimator=GeometricEstimator(), redetect_after=2)
    flags = [est.process(make_frame(scene, seed=i)).tracked for i in range(6)]
    assert flags == [False, True, True, False, True, True]


def test_reset_forces_re_registration():
    scene = SyntheticScene(forward_m=1.6)
    est = DepthOnlyFoundationPose(estimator=GeometricEstimator(), redetect_after=0)
    assert est.process(make_frame(scene)).tracked is False
    assert est.process(make_frame(scene)).tracked is True
    est.reset()
    assert est.process(make_frame(scene)).tracked is False


def test_registration_returns_none_when_there_is_nothing_to_see():
    frame = Frame(depth_m=np.zeros((120, 160), np.float32), intrinsics=X2_INTR)
    est = DepthOnlyFoundationPose(estimator=GeometricEstimator())
    assert est.process(frame) is None


def test_distance_tracks_a_receding_box():
    """The reported distance must move the right way and by the right amount."""
    est = DepthOnlyFoundationPose(extents_m=[0.40, 0.30, 0.25], estimator=GeometricEstimator())
    measured, truth = [], []
    for d in (1.0, 1.5, 2.0, 2.5):
        scene = SyntheticScene(forward_m=d)
        est.reset()
        result = est.process(make_frame(scene))
        assert result is not None
        measured.append(result.distance_m)
        truth.append(float(np.linalg.norm(closest_point_on_box(scene.truth_box()))))

    assert measured == sorted(measured)
    assert np.allclose(measured, truth, atol=0.08)


def test_missing_foundationpose_raises_an_actionable_error():
    est = DepthOnlyFoundationPose(extents_m=[0.4, 0.3, 0.25])
    with pytest.raises(ImportError, match="FoundationPose"):
        est._build_estimator()


# --------------------------------------------------------------------------
# Camera setup
#
# The intrinsics presets and the X2's ROS 2 depth interface live in
# test_cameras.py. Only the interaction with this engine belongs here.
# --------------------------------------------------------------------------


def test_the_pipeline_runs_at_the_x2_cameras_intrinsics():
    """The default detector is tuned on a D435; check it holds at 90 deg / 1280x800."""
    from boxrange.pipeline import BoxRangePipeline

    scene = SyntheticScene(forward_m=2.0)
    frame = make_frame(scene, intr=gemini335_depth())
    detections = BoxRangePipeline().process(frame)
    assert detections, "no box found with the Gemini 335 intrinsics"

    truth = float(np.linalg.norm(closest_point_on_box(scene.truth_box())))
    assert detections[0].range.surface_m == pytest.approx(truth, abs=0.10)
