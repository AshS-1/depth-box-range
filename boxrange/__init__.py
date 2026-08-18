"""boxrange -- measure the distance to a box from depth-camera input.

Quick start::

    from boxrange import BoxRangePipeline, SyntheticSource

    pipeline = BoxRangePipeline()
    with SyntheticSource() as source:
        for frame in source:
            for det in pipeline.process(frame):
                print(det.range)

Swap ``SyntheticSource`` for ``X2RgbdSource()`` (the AgiBot X2's head RGB-D
camera, over the robot's ROS 2 interface), ``OrbbecSource()`` (a Gemini 335
straight over USB) or ``RealSenseSource()`` to run against hardware; the
pipeline itself does not change.

:mod:`boxrange.foundationpose` is an alternative engine that drives NVlabs
FoundationPose from depth alone. It needs CUDA and buys you full 6D pose of a
known object rather than a better distance -- read that module's docstring
before reaching for it.
"""

from __future__ import annotations

from .foundationpose import (
    DepthOnlyFoundationPose,
    PoseResult,
    depth_to_pseudo_rgb,
    surface_normals,
)
from .frames import (
    X2_DEPTH_IMAGE,
    X2_DEPTH_INFO,
    Frame,
    FrameSource,
    NpzSource,
    OrbbecSource,
    RealSenseSource,
    SyntheticScene,
    SyntheticSource,
    X2RgbdSource,
    decode_depth_image,
    intrinsics_from_camera_info,
    record_npz,
    render_depth,
)
from .geometry import (
    OrientedBox,
    Plane,
    count_visible_faces,
    deproject,
    fit_box_on_plane,
    fit_plane_ransac,
)
from .intrinsics import (
    CAMERA_PRESETS,
    PRESET_RESOLUTIONS,
    CameraIntrinsics,
    d435_depth,
    gemini335_depth,
)
from .pipeline import BoxDetection, BoxRangePipeline, detection_to_dict
from .ranging import RangeEstimate, closest_point_on_box, estimate_range
from .segment import Candidate, Detector, PlaneClusterDetector, RegionPriorDetector

__version__ = "0.1.0"

__all__ = [
    "CAMERA_PRESETS",
    "PRESET_RESOLUTIONS",
    "X2_DEPTH_IMAGE",
    "X2_DEPTH_INFO",
    "BoxDetection",
    "BoxRangePipeline",
    "CameraIntrinsics",
    "Candidate",
    "DepthOnlyFoundationPose",
    "Detector",
    "Frame",
    "FrameSource",
    "NpzSource",
    "OrbbecSource",
    "OrientedBox",
    "Plane",
    "PlaneClusterDetector",
    "PoseResult",
    "RangeEstimate",
    "RealSenseSource",
    "RegionPriorDetector",
    "SyntheticScene",
    "SyntheticSource",
    "X2RgbdSource",
    "__version__",
    "closest_point_on_box",
    "count_visible_faces",
    "d435_depth",
    "decode_depth_image",
    "deproject",
    "depth_to_pseudo_rgb",
    "detection_to_dict",
    "estimate_range",
    "fit_box_on_plane",
    "fit_plane_ransac",
    "gemini335_depth",
    "intrinsics_from_camera_info",
    "record_npz",
    "render_depth",
    "surface_normals",
]
