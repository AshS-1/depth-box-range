"""boxrange -- measure the distance to a box from RealSense depth.

Quick start::

    from boxrange import BoxRangePipeline, SyntheticSource

    pipeline = BoxRangePipeline()
    with SyntheticSource() as source:
        for frame in source:
            for det in pipeline.process(frame):
                print(det.range)

Swap ``SyntheticSource`` for ``RealSenseSource()`` to run against hardware; the
pipeline itself does not change.
"""

from __future__ import annotations

from .frames import (
    Frame,
    FrameSource,
    NpzSource,
    RealSenseSource,
    SyntheticScene,
    SyntheticSource,
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
from .intrinsics import CameraIntrinsics
from .pipeline import BoxDetection, BoxRangePipeline, detection_to_dict
from .ranging import RangeEstimate, closest_point_on_box, estimate_range
from .segment import Candidate, Detector, PlaneClusterDetector, RegionPriorDetector

__version__ = "0.1.0"

__all__ = [
    "BoxDetection",
    "BoxRangePipeline",
    "CameraIntrinsics",
    "Candidate",
    "Detector",
    "Frame",
    "FrameSource",
    "NpzSource",
    "OrientedBox",
    "Plane",
    "PlaneClusterDetector",
    "RangeEstimate",
    "RealSenseSource",
    "RegionPriorDetector",
    "SyntheticScene",
    "SyntheticSource",
    "__version__",
    "closest_point_on_box",
    "count_visible_faces",
    "deproject",
    "detection_to_dict",
    "estimate_range",
    "fit_box_on_plane",
    "fit_plane_ransac",
    "record_npz",
    "render_depth",
]
