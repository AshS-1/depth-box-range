"""Camera setup: intrinsics presets, and the AgiBot X2's ROS 2 depth interface.

The X2 path is split so that everything except the socket is testable here. The
message decoding and the CameraInfo conversion are pure functions over bytes and
plain attributes, so they run with no ROS 2 installed; only :class:`X2RgbdSource`
itself needs rclpy, and it is a thin wrapper over the two.

Depth-unit conversions get disproportionate attention on purpose. Every one of
them fails by a factor of 1000, and because the pipeline's thresholds all scale
with the noise model the symptom is an empty detection list rather than an
obviously wrong distance.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxrange.frames import (
    X2_DEPTH_IMAGE,
    decode_depth_image,
    depth_quantisation,
    intrinsics_from_camera_info,
)
from boxrange.intrinsics import (
    CAMERA_PRESETS,
    PRESET_RESOLUTIONS,
    CameraIntrinsics,
    gemini335_depth,
)


class FakeCameraInfo:
    """The three fields of ``sensor_msgs/CameraInfo`` this package reads."""

    def __init__(self, fx=640.0, fy=640.0, cx=639.5, cy=399.5, width=1280, height=800,
                 attr="k"):
        self.width, self.height = width, height
        setattr(self, attr, [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# sensor_msgs/Image decoding
# --------------------------------------------------------------------------


def test_16uc1_depth_is_millimetres():
    """The X2 docs give the topic and type but never the encoding.

    ROS publishes depth as 16UC1 in millimetres or 32FC1 in metres, so the
    encoding field has to be read rather than assumed.
    """
    raw = np.array([[0, 1000], [2500, 65535]], dtype="<u2")
    out = decode_depth_image(raw.tobytes(), 2, 2, "16UC1", step=4)
    assert out.dtype == np.float32
    assert np.allclose(out, [[0.0, 1.0], [2.5, 65.535]])


def test_32fc1_depth_is_already_metres():
    raw = np.array([[0.0, 1.0], [2.5, 4.0]], dtype="<f4")
    out = decode_depth_image(raw.tobytes(), 2, 2, "32FC1", step=8)
    assert np.allclose(out, [[0.0, 1.0], [2.5, 4.0]])


def test_mono16_is_treated_as_millimetre_depth():
    raw = np.array([[1500]], dtype="<u2")
    assert decode_depth_image(raw.tobytes(), 1, 1, "mono16", step=2)[0, 0] == pytest.approx(1.5)


def test_row_padding_is_honoured():
    """``step`` is a byte stride and is not always ``width * itemsize``.

    Reshaping by width alone shears the image when rows are padded, which looks
    like a wildly tilted floor rather than like a bug -- and the plane fit will
    happily fit that tilted floor and report nonsense with high confidence.
    """
    width, height, pad = 3, 2, 2
    rows = np.array([[1000, 2000, 3000, 111, 222], [4000, 5000, 6000, 333, 444]], dtype="<u2")
    step = (width + pad) * 2
    out = decode_depth_image(rows.tobytes(), height, width, "16UC1", step=step)
    assert out.shape == (height, width)
    assert np.allclose(out, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_big_endian_payloads_are_byte_swapped():
    raw = np.array([[1000, 2000]], dtype=">u2")
    out = decode_depth_image(raw.tobytes(), 1, 2, "16UC1", step=4, is_bigendian=True)
    assert np.allclose(out, [[1.0, 2.0]])


def test_non_finite_and_negative_depth_becomes_the_no_return_marker():
    """0 is this package's "no return"; NaN and inf would poison every fit."""
    raw = np.array([[np.nan, np.inf, -1.0, 2.0]], dtype="<f4")
    out = decode_depth_image(raw.tobytes(), 1, 4, "32FC1", step=16)
    assert np.array_equal(out, np.array([[0.0, 0.0, 0.0, 2.0]], dtype=np.float32))
    assert np.isfinite(out).all()


def test_unknown_encoding_is_refused_by_name():
    """Better to stop than to guess: guessing wrong is a factor of 1000."""
    with pytest.raises(ValueError, match="16UC1"):
        decode_depth_image(b"\x00" * 8, 2, 2, "rgb8", step=4)


def test_too_small_a_step_is_refused():
    with pytest.raises(ValueError, match="step"):
        decode_depth_image(b"\x00" * 8, 2, 4, "16UC1", step=4)


def test_quantisation_reflects_the_encoding():
    """A millimetre floor is real on 16UC1 and absent on 32FC1.

    It feeds the reported sigma, so claiming one on float depth would inflate
    every uncertainty by a millimetre for no reason.
    """
    assert depth_quantisation("16UC1") == pytest.approx(1e-3)
    assert depth_quantisation("mono16") == pytest.approx(1e-3)
    assert depth_quantisation("32FC1") == 0.0


# --------------------------------------------------------------------------
# sensor_msgs/CameraInfo
# --------------------------------------------------------------------------


def test_camera_info_gives_the_per_device_calibration():
    intr = intrinsics_from_camera_info(FakeCameraInfo(fx=612.5, fy=612.5, cx=636.0, cy=402.0))
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (612.5, 612.5, 636.0, 402.0)
    assert (intr.width, intr.height) == (1280, 800)
    # CameraInfo carries no baseline, so the noise model still needs the datasheet.
    assert intr.baseline_m == pytest.approx(gemini335_depth().baseline_m)


def test_camera_info_accepts_the_ros1_spelling():
    """rclpy exposes the matrix as `k`; ROS 1 and some bridges use `K`."""
    intr = intrinsics_from_camera_info(FakeCameraInfo(fx=600.0, attr="K"))
    assert intr.fx == 600.0


def test_camera_info_without_intrinsics_is_refused():
    """Zeroed intrinsics mean images are flowing before calibration loaded.

    Deprojecting with fx=0 divides by zero and yields an all-NaN cloud, which
    surfaces as "no boxes found" -- indistinguishable from an empty room.
    """
    with pytest.raises(ValueError, match="no usable intrinsics"):
        intrinsics_from_camera_info(FakeCameraInfo(fx=0.0, fy=0.0))

    class Bare:
        width = height = 4

    with pytest.raises(ValueError, match="neither"):
        intrinsics_from_camera_info(Bare())


def test_camera_info_overrides_apply():
    intr = intrinsics_from_camera_info(FakeCameraInfo(), depth_scale=1e-3)
    assert intr.depth_scale == pytest.approx(1e-3)


def test_the_x2_depth_topic_is_the_head_camera():
    """The RGB-D camera is in the head. The chest carries the LiDAR and an IMU.

    Pinned because the sensor page does not state mount locations and it is easy
    to infer the wrong one from the hardware list.
    """
    assert X2_DEPTH_IMAGE == "/aima/hal/sensor/rgbd_head_front/depth_image"


# --------------------------------------------------------------------------
# Intrinsics presets
# --------------------------------------------------------------------------


def test_gemini335_preset_reproduces_the_datasheet():
    intr = gemini335_depth()
    assert (intr.width, intr.height) == (1280, 800)
    # 90 deg horizontal FOV.
    assert np.rad2deg(2 * np.arctan(intr.width / 2 / intr.fx)) == pytest.approx(90.0)
    # Vertical follows from square pixels and lands inside the quoted 65 +/- 3.
    vfov = np.rad2deg(2 * np.arctan(intr.height / 2 / intr.fy))
    assert 62.0 <= vfov <= 68.0
    # Spatial precision <= 1.5% at 2 m.
    assert intr.range_sigma(2.0) == pytest.approx(0.015 * 2.0, rel=1e-6)


def test_presets_keep_their_native_aspect_ratio():
    """Forcing the Gemini to 4:3 would model a vertical FOV it does not have."""
    for name, (w, h) in PRESET_RESOLUTIONS.items():
        intr = CAMERA_PRESETS[name](w, h)
        assert (intr.width, intr.height) == (w, h)
    gem = CAMERA_PRESETS["gemini335"](*PRESET_RESOLUTIONS["gemini335"])
    assert np.rad2deg(2 * np.arctan(gem.width / 2 / gem.fx)) == pytest.approx(90.0)
    assert 62.0 <= np.rad2deg(2 * np.arctan(gem.height / 2 / gem.fy)) <= 68.0


def test_from_fov_round_trips_and_validates():
    intr = CameraIntrinsics.from_fov(848, 480, hfov_deg=87.0, vfov_deg=58.0)
    assert np.rad2deg(2 * np.arctan(848 / 2 / intr.fx)) == pytest.approx(87.0)
    assert np.rad2deg(2 * np.arctan(480 / 2 / intr.fy)) == pytest.approx(58.0)
    with pytest.raises(ValueError):
        CameraIntrinsics.from_fov(640, 480, hfov_deg=200.0)
    with pytest.raises(ValueError):
        CameraIntrinsics.from_fov(640, 480, hfov_deg=90.0, vfov_deg=0.0)


def test_from_orbbec_converts_millimetres_to_metres():
    """Orbbec's get_depth_scale() is mm/unit; RealSense's is m/unit.

    Mixing them is a factor of 1000 that shows up as an empty detection list
    rather than an obviously wrong distance, so it is worth a test of its own.
    """

    class FakeIntrinsic:
        width, height = 1280, 800
        fx = fy = 640.0
        cx, cy = 639.5, 399.5

    assert CameraIntrinsics.from_orbbec(FakeIntrinsic(), 1.0).depth_scale == pytest.approx(1e-3)
    assert CameraIntrinsics.from_orbbec(FakeIntrinsic(), 0.1).depth_scale == pytest.approx(1e-4)


# --------------------------------------------------------------------------
# Record / replay
# --------------------------------------------------------------------------


def test_recording_round_trips_the_whole_noise_model(tmp_path):
    """Replay must reproduce the capture sensor, not just its pixels.

    ``subpixel_px`` is what makes range_sigma sensor-specific, and every
    threshold in the pipeline scales with range_sigma. Dropping it replays a
    Gemini 335 capture under a D435's disparity noise: same depth, quietly
    different clustering tolerance, confidence and reported uncertainty.
    """
    from boxrange.frames import NpzSource, SyntheticScene, SyntheticSource, record_npz

    intr = gemini335_depth(640, 400)
    path = tmp_path / "run.npz"
    record_npz(SyntheticSource(SyntheticScene(), intr, frames=2), path)

    replayed = NpzSource(path).intrinsics
    assert replayed == intr
    assert replayed.range_sigma(3.0) == pytest.approx(intr.range_sigma(3.0))


def test_legacy_recordings_still_load(tmp_path):
    """A recording made before subpixel_px was saved was made on a D435."""
    from boxrange.frames import NpzSource

    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        depth_m=np.ones((1, 4, 4), dtype=np.float32),
        width=4, height=4, fx=385.0, fy=385.0, cx=1.5, cy=1.5,
    )
    assert NpzSource(path).intrinsics.subpixel_px == pytest.approx(0.08)
