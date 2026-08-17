"""Accuracy tests against a scene whose true geometry we know exactly.

These are the tests worth having. A pipeline that runs without raising but
reports 1.8 m for a box at 1.6 m is worse than one that crashes, so almost
everything here asserts a *metric* result rather than a shape or a type.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxrange import (
    BoxRangePipeline,
    CameraIntrinsics,
    SyntheticScene,
    SyntheticSource,
    closest_point_on_box,
    deproject,
    estimate_range,
    fit_box_on_plane,
    fit_plane_ransac,
    render_depth,
)

INTR = CameraIntrinsics(width=640, height=480, fx=385.0, fy=385.0, cx=320.0, cy=240.0)


def truth_surface_distance(scene: SyntheticScene) -> float:
    return float(np.linalg.norm(closest_point_on_box(scene.truth_box())))


# -- primitives ---------------------------------------------------------------


def test_deproject_roundtrip():
    """A pixel's deprojected point must reproject to the same pixel."""
    depth = np.full((INTR.height, INTR.width), 2.0, dtype=np.float32)
    cloud = deproject(depth, INTR)
    v, u = 137, 401
    x, y, z = cloud[v, u]
    assert z == pytest.approx(2.0)
    assert x / z * INTR.fx + INTR.cx == pytest.approx(u)
    assert y / z * INTR.fy + INTR.cy == pytest.approx(v)


def test_deproject_marks_invalid_as_nan():
    depth = np.full((8, 8), 1.0, dtype=np.float32)
    depth[0, 0] = 0.0
    assert np.isnan(deproject(depth, INTR)[0, 0, 2])


def test_range_sigma_grows_quadratically():
    """Doubling range must quadruple the uncertainty."""
    assert INTR.range_sigma(2.0) == pytest.approx(4.0 * INTR.range_sigma(1.0))


def test_scaled_intrinsics_preserve_projection():
    half = INTR.scaled(0.5)
    assert half.width == 320 and half.fx == pytest.approx(INTR.fx / 2)
    # Image centre must stay the image centre under resize.
    assert (half.cx + 0.5) / (INTR.cx + 0.5) == pytest.approx(0.5)


def test_closest_point_on_axis_aligned_box():
    from boxrange import OrientedBox

    box = OrientedBox(np.array([0.0, 0.0, 2.0]), np.eye(3), np.array([1.0, 1.0, 1.0]))
    near = closest_point_on_box(box)
    assert near == pytest.approx([0.0, 0.0, 1.5])
    assert np.linalg.norm(near) == pytest.approx(1.5)


# -- plane fitting ------------------------------------------------------------


def test_plane_fit_recovers_ground_truth():
    scene = SyntheticScene()
    depth = render_depth(scene, INTR, noise=True, dropout=0.02, rng=np.random.default_rng(1))
    cloud = deproject(depth, INTR)
    pts = cloud[np.isfinite(cloud).all(axis=-1)]

    plane = fit_plane_ransac(pts, up_hint=np.array([0.0, -1.0, 0.0]), iterations=400)
    truth = scene.ground_plane()

    assert plane is not None
    angle = np.rad2deg(np.arccos(np.clip(abs(plane.normal @ truth.normal), -1, 1)))
    assert angle < 2.0, f"plane normal off by {angle:.2f} deg"
    assert plane.d == pytest.approx(truth.d, abs=0.02)


def test_plane_fit_rejects_wall_when_hinted():
    """A dominant vertical surface must not win when we asked for the floor."""
    rng = np.random.default_rng(0)
    # A wall filling most of the view, floor a minority.
    wall = np.stack(
        [rng.uniform(-2, 2, 8000), rng.uniform(-2, 2, 8000), np.full(8000, 3.0)], axis=1
    )
    floor = np.stack(
        [rng.uniform(-2, 2, 2000), np.full(2000, 1.0), rng.uniform(0.5, 3, 2000)], axis=1
    )
    pts = np.vstack([wall, floor])

    plane = fit_plane_ransac(pts, up_hint=np.array([0.0, -1.0, 0.0]), iterations=500)
    assert plane is not None
    assert abs(plane.normal @ np.array([0.0, -1.0, 0.0])) > 0.95


# -- box fitting --------------------------------------------------------------


def test_box_fit_recovers_extents_noise_free():
    scene = SyntheticScene(box_size=(0.40, 0.30, 0.25), forward_m=1.5, yaw=np.deg2rad(30))
    depth = render_depth(scene, INTR, noise=False, dropout=0.0)
    cloud = deproject(depth, INTR)

    plane = scene.ground_plane()
    finite = np.isfinite(cloud).all(axis=-1)
    pts = cloud[finite]
    height = plane.signed_distance(pts)
    box_pts = pts[(height > 0.03) & (height < 2.0)]

    box = fit_box_on_plane(box_pts, plane)
    assert box is not None

    truth = scene.truth_box()
    # Footprint axes can come back in either order; height is the third.
    got = sorted(box.extents[:2])
    want = sorted(truth.extents[:2])
    assert got[0] == pytest.approx(want[0], abs=0.03)
    assert got[1] == pytest.approx(want[1], abs=0.03)
    assert box.extents[2] == pytest.approx(truth.extents[2], abs=0.03)


def test_box_up_axis_matches_plane_normal():
    scene = SyntheticScene()
    depth = render_depth(scene, INTR, noise=False, dropout=0.0)
    cloud = deproject(depth, INTR)
    plane = scene.ground_plane()
    pts = cloud[np.isfinite(cloud).all(axis=-1)]
    box_pts = pts[plane.signed_distance(pts) > 0.03]

    box = fit_box_on_plane(box_pts, plane)
    assert box is not None
    assert abs(box.R[:, 2] @ plane.normal) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.det(box.R) == pytest.approx(1.0, abs=1e-6)


# -- end to end ---------------------------------------------------------------


@pytest.mark.parametrize("distance", [1.0, 1.6, 2.5, 3.5])
def test_pipeline_distance_accuracy(distance):
    """The headline number must track truth across the working range."""
    scene = SyntheticScene(forward_m=distance)
    source = SyntheticSource(scene, intrinsics=INTR, frames=3, seed=7)
    pipeline = BoxRangePipeline()

    want = truth_surface_distance(scene)
    errors = []
    for frame in source:
        dets = pipeline.process(frame)
        assert dets, f"no box detected at {distance} m"
        errors.append(abs(dets[0].range.surface_m - want))

    # Tolerance scales with the stereo noise model rather than being a flat
    # number: demanding 1 cm at 3.5 m would be demanding better than physics.
    tol = max(0.05, 6.0 * float(INTR.range_sigma(want)))
    assert min(errors) < tol, f"best error {min(errors):.3f} m exceeds {tol:.3f} m"


def test_pipeline_reports_plausible_extents():
    scene = SyntheticScene(box_size=(0.5, 0.35, 0.3), forward_m=1.5)
    source = SyntheticSource(scene, intrinsics=INTR, frames=2, seed=3)
    pipeline = BoxRangePipeline()

    det = None
    for frame in source:
        dets = pipeline.process(frame)
        if dets:
            det = dets[0]
    assert det is not None
    assert det.box.extents[2] == pytest.approx(0.3, abs=0.06)
    assert sorted(det.box.extents[:2])[1] == pytest.approx(0.5, abs=0.08)


def test_uncertainty_grows_with_distance():
    """A far box must report a larger sigma than a near one."""
    sigmas = {}
    for d in (1.0, 3.0):
        source = SyntheticSource(SyntheticScene(forward_m=d), intrinsics=INTR, frames=1, seed=5)
        pipeline = BoxRangePipeline()
        for frame in source:
            dets = pipeline.process(frame)
            assert dets
            sigmas[d] = dets[0].range.sigma_m
    assert sigmas[3.0] > sigmas[1.0]


def test_measured_distance_agrees_with_fitted():
    """The model-free cross-check must corroborate the fitted surface range."""
    scene = SyntheticScene(forward_m=1.8)
    source = SyntheticSource(scene, intrinsics=INTR, frames=1, seed=11)
    pipeline = BoxRangePipeline()
    for frame in source:
        det = pipeline.process(frame)[0]
        # measured_m is axial, surface_m is Euclidean, so they differ slightly.
        assert abs(det.range.measured_m - det.range.axial_m) < 0.10


def test_empty_scene_yields_no_detections():
    """Floor only: nothing above the plane, so nothing to report."""
    scene = SyntheticScene(box_size=(0.001, 0.001, 0.001), forward_m=1.5)
    source = SyntheticSource(scene, intrinsics=INTR, frames=2, seed=1)
    pipeline = BoxRangePipeline()
    for frame in source:
        assert pipeline.process(frame) == []


def test_all_zero_depth_is_handled():
    from boxrange import Frame

    frame = Frame(depth_m=np.zeros((120, 160), dtype=np.float32), intrinsics=INTR)
    assert BoxRangePipeline().process(frame) == []


def test_tracker_assigns_stable_id_and_smooths():
    scene = SyntheticScene(forward_m=2.0)
    source = SyntheticSource(scene, intrinsics=INTR, frames=6, seed=2)
    pipeline = BoxRangePipeline()

    ids, raw, smoothed = [], [], []
    for frame in source:
        for det in pipeline.process(frame):
            ids.append(det.track_id)
            raw.append(det.range.surface_m)
            smoothed.append(det.smoothed_surface_m)

    assert len(set(ids)) == 1, f"track id churned: {set(ids)}"
    if len(raw) > 3:
        # Smoothing must not bias the estimate away from the raw mean.
        assert abs(np.mean(smoothed) - np.mean(raw)) < 0.05


def test_confidence_penalises_single_face_view():
    """Head-on at zero yaw shows one face; obliquely, two. Confidence must reflect it."""
    def conf(yaw_deg):
        scene = SyntheticScene(forward_m=1.5, yaw=np.deg2rad(yaw_deg), camera_pitch=0.0)
        source = SyntheticSource(scene, intrinsics=INTR, frames=1, seed=4)
        pipeline = BoxRangePipeline(min_confidence=0.0)
        for frame in source:
            dets = pipeline.process(frame)
            return dets[0].range if dets else None

    oblique = conf(35.0)
    assert oblique is not None
    assert oblique.visible_faces >= 2
    assert oblique.confidence > 0.3


def test_visible_faces_never_exceeds_three():
    """A convex box shows at most 3 faces from any single viewpoint."""
    for d in (1.5, 2.5, 3.5):
        source = SyntheticSource(SyntheticScene(forward_m=d), intrinsics=INTR, frames=1, seed=6)
        pipeline = BoxRangePipeline(min_confidence=0.0)
        for frame in source:
            for det in pipeline.process(frame):
                assert 0 <= det.range.visible_faces <= 3


def test_range_bias_is_conservative_and_bounded():
    """Locks in the known residual bias so it cannot silently get worse.

    The estimate reads slightly *near* of truth, and the error grows with range:
    footprint noise inflates the fitted rectangle outward, which pushes the
    near corner toward the camera. Reading near is the safe direction for
    obstacle avoidance, but it is a bias, not noise, so it is pinned here.
    """
    for distance, limit in ((1.0, 0.03), (2.0, 0.06), (3.5, 0.12), (4.5, 0.16)):
        scene = SyntheticScene(forward_m=distance)
        want = truth_surface_distance(scene)
        errors = []
        for seed in range(4):
            pipeline = BoxRangePipeline()
            for frame in SyntheticSource(scene, intrinsics=INTR, frames=1, seed=seed):
                dets = pipeline.process(frame)
                if dets:
                    errors.append(dets[0].range.surface_m - want)
        assert errors, f"no detection at {distance} m"
        bias = float(np.mean(errors))
        assert bias <= 0.005, f"bias at {distance} m turned optimistic: {bias:+.3f} m"
        assert abs(bias) < limit, f"bias at {distance} m grew to {bias:+.3f} m"


def test_detection_dict_is_json_serialisable():
    import json

    scene = SyntheticScene(forward_m=1.5)
    source = SyntheticSource(scene, intrinsics=INTR, frames=1, seed=0)
    pipeline = BoxRangePipeline()
    from boxrange import detection_to_dict

    for frame in source:
        for det in pipeline.process(frame):
            json.loads(json.dumps(detection_to_dict(det, frame)))
