"""Edge cases and regression tests for bugs that were actually hit.

Each test here corresponds to a concrete failure found by throwing degenerate
input at the pipeline, not to a hypothetical one.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

from boxrange import (
    BoxRangePipeline,
    CameraIntrinsics,
    Frame,
    NpzSource,
    PlaneClusterDetector,
    RegionPriorDetector,
    SyntheticScene,
    SyntheticSource,
    deproject,
    estimate_range,
    fit_box_on_plane,
    record_npz,
    render_depth,
)
from boxrange.pipeline import BoxDetection, detection_to_dict
from boxrange.viz import project, render_frame

INTR = CameraIntrinsics(width=640, height=480, fx=385.0, fy=385.0, cx=320.0, cy=240.0)
SMALL = CameraIntrinsics(width=64, height=48, fx=38.0, fy=38.0, cx=32.0, cy=24.0)


def _detection(distance: float = 2.0):
    depth = render_depth(SyntheticScene(forward_m=distance), INTR, rng=np.random.default_rng(0))
    cands, plane = PlaneClusterDetector().detect(depth, INTR)
    box = fit_box_on_plane(cands[0].points, plane)
    rng_est = estimate_range(cands[0].points, box, INTR, plane=plane)
    return depth, BoxDetection(box=box, range=rng_est, candidate=cands[0], plane=plane)


# -- degenerate depth ---------------------------------------------------------


@pytest.mark.parametrize(
    "name,depth",
    [
        ("zeros", np.zeros((48, 64), np.float32)),
        ("nan", np.full((48, 64), np.nan, np.float32)),
        ("inf", np.full((48, 64), np.inf, np.float32)),
        ("negative", np.full((48, 64), -1.0, np.float32)),
        ("huge", np.full((48, 64), 1e9, np.float32)),
        ("tiny", np.full((48, 64), 1e-9, np.float32)),
        ("flat_plane", np.full((48, 64), 2.0, np.float32)),
        ("mixed_nonfinite", np.where(
            np.arange(48 * 64).reshape(48, 64) % 2 == 0, np.nan, np.inf).astype(np.float32)),
    ],
)
def test_degenerate_depth_never_raises(name, depth):
    """Garbage in must produce no detections, not an exception."""
    assert BoxRangePipeline().process(Frame(depth_m=depth, intrinsics=SMALL)) == []


def test_single_pixel_image():
    frame = Frame(depth_m=np.full((1, 1), 2.0, np.float32),
                  intrinsics=CameraIntrinsics(1, 1, 38.0, 38.0, 0.5, 0.5))
    assert BoxRangePipeline().process(frame) == []


def test_deproject_emits_no_warnings_on_nonfinite():
    """0 * inf at the principal point used to raise an invalid-value warning."""
    depth = np.full((16, 16), np.inf, np.float32)
    depth[8, 8] = 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cloud = deproject(depth, CameraIntrinsics(16, 16, 10.0, 10.0, 8.0, 8.0))
    assert np.isfinite(cloud[8, 8]).all()


# -- JSON output --------------------------------------------------------------


def test_json_never_contains_nan_or_infinity():
    """An untracked detection carries a NaN smoothed range -> invalid JSON."""
    depth, det = _detection()
    frame = Frame(depth_m=depth, intrinsics=INTR)
    assert math.isnan(det.smoothed_surface_m), "precondition: this det is untracked"

    text = json.dumps(detection_to_dict(det, frame))
    assert "NaN" not in text and "Infinity" not in text
    # A strict parser must accept it.
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite constant: {c}"))


def test_json_untracked_smoothed_is_null():
    depth, det = _detection()
    record = detection_to_dict(det, Frame(depth_m=depth, intrinsics=INTR))
    assert record["distance_smoothed_m"] is None
    assert record["distance_m"] is not None


# -- visualisation ------------------------------------------------------------


def test_project_far_offscreen_point_does_not_crash_drawing():
    """Coordinates near 1e13 are finite but overflow int32 inside cv2.line."""
    depth, det = _detection()
    # A box practically on the image plane throws corners to ~1e13 px.
    from boxrange.geometry import OrientedBox

    exploded = BoxDetection(
        box=OrientedBox(np.array([0.0, 0.0, 1e-6]), np.eye(3), np.array([1.0, 1.0, 1.0])),
        range=det.range,
        candidate=det.candidate,
        plane=det.plane,
    )
    out = render_frame(depth, [exploded], INTR)
    assert out.shape[:2] == depth.shape


def test_project_marks_points_behind_camera_invalid():
    uv = project(np.array([[1.0, 1.0, -2.0], [1.0, 1.0, 0.0]]), INTR)
    assert np.isnan(uv).all()


def test_render_frame_with_no_detections():
    depth = render_depth(SyntheticScene(), INTR, rng=np.random.default_rng(0))
    assert render_frame(depth, [], INTR).shape == (INTR.height, INTR.width, 3)


# -- region prior adapter -----------------------------------------------------


def test_region_shape_mismatch_raises_clear_error():
    depth = render_depth(SyntheticScene(forward_m=2.0), INTR, rng=np.random.default_rng(0))
    det = RegionPriorDetector(inner=PlaneClusterDetector())
    with pytest.raises(ValueError, match="region 0 has shape"):
        det.detect_with_regions(depth, INTR, [np.ones((10, 10), bool)])


def test_region_prior_full_frame_matches_plain_detection():
    depth = render_depth(SyntheticScene(forward_m=2.0), INTR, rng=np.random.default_rng(0))
    plain, _ = PlaneClusterDetector().detect(depth, INTR)
    regional, _ = RegionPriorDetector(inner=PlaneClusterDetector()).detect_with_regions(
        depth, INTR, [np.ones(depth.shape, bool)]
    )
    assert len(regional) == 1
    assert regional[0].size == plain[0].size


def test_region_prior_empty_region_yields_nothing():
    depth = render_depth(SyntheticScene(forward_m=2.0), INTR, rng=np.random.default_rng(0))
    out, _ = RegionPriorDetector(inner=PlaneClusterDetector()).detect_with_regions(
        depth, INTR, [np.zeros(depth.shape, bool)]
    )
    assert out == []


# -- parameter validation -----------------------------------------------------


@pytest.mark.parametrize("trim", [-1.0, 50.0, 80.0])
def test_invalid_trim_percent_rejected(trim):
    depth = render_depth(SyntheticScene(forward_m=2.0), INTR, rng=np.random.default_rng(0))
    cands, plane = PlaneClusterDetector().detect(depth, INTR)
    with pytest.raises(ValueError, match="trim_percent"):
        fit_box_on_plane(cands[0].points, plane, trim_percent=trim)


# -- recording round trip -----------------------------------------------------


def test_npz_roundtrip_preserves_intrinsics_and_ranging(tmp_path):
    path = tmp_path / "run.npz"
    scene = SyntheticScene(forward_m=2.0)
    n = record_npz(SyntheticSource(scene, intrinsics=INTR, frames=3, seed=0), path, max_frames=3)
    assert n == 3

    source = NpzSource(path)
    assert source.intrinsics.fx == pytest.approx(INTR.fx)
    assert source.intrinsics.depth_scale == pytest.approx(INTR.depth_scale)
    assert source.intrinsics.baseline_m == pytest.approx(INTR.baseline_m)

    pipeline = BoxRangePipeline()
    found = [d for frame in source for d in pipeline.process(frame)]
    assert found, "replayed frames must still yield detections"


def test_npz_accepts_single_frame_saved_2d(tmp_path):
    path = tmp_path / "one.npz"
    depth = render_depth(SyntheticScene(forward_m=2.0), INTR, rng=np.random.default_rng(0))
    np.savez_compressed(
        path, depth_m=depth, width=INTR.width, height=INTR.height,
        fx=INTR.fx, fy=INTR.fy, cx=INTR.cx, cy=INTR.cy,
    )
    assert len(list(NpzSource(path))) == 1


# -- pipeline state -----------------------------------------------------------

def test_reset_clears_track_ids():
    scene = SyntheticScene(forward_m=2.0)
    pipeline = BoxRangePipeline()
    for frame in SyntheticSource(scene, intrinsics=INTR, frames=2, seed=0):
        pipeline.process(frame)
    pipeline.reset()
    for frame in SyntheticSource(scene, intrinsics=INTR, frames=1, seed=0):
        for det in pipeline.process(frame):
            assert det.track_id == 0


def test_stale_tracks_are_dropped():
    """A box that leaves the scene must not keep its slot forever."""
    pipeline = BoxRangePipeline(max_misses=2)
    for frame in SyntheticSource(SyntheticScene(forward_m=2.0), intrinsics=INTR, frames=2, seed=0):
        pipeline.process(frame)
    blank = Frame(depth_m=np.zeros((480, 640), np.float32), intrinsics=INTR)
    for _ in range(5):
        pipeline.process(blank)
    assert pipeline._tracks == []
