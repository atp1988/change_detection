import os
from dataclasses import dataclass, field
from pathlib import Path


# If you use ROI GUI selection on Linux:
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


# ── Paths (project-root relative) ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../change_det_V1
MODELS_DIR = PROJECT_ROOT / "weights"

# Must exist/be creatable at runtime in project root:
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "compares" / "runs")

# Default ROI file path — relative to the execution directory (kept like before)
DEFAULT_ROI_FILE = "/home/atp/projects/change_det_V1/roi.json"


_DEFAULT_OFFICE_CLASSES = [
    "ceramic plate", "dinner plate",
    "monitor", "laptop",
    "keyboard",
    "computer box",
    "smartphone",
    "cup",
    "trash can", "tissue box",
    "book", "notebook",
]


@dataclass
class DetectorConfig:
    # YOLO
    yolo_conf: float = 0.18
    yolo_iou: float = 0.45
    yolo_model: str = str(MODELS_DIR / "yolov8l-worldv2.pt")  # default in /models

    # Matching / thresholds
    similarity_threshold: float = 0.65
    double_check_threshold: float = 0.50
    position_tolerance_px: int = 100
    alpha: float = 0.25

    # Cropping / embedding
    crop_padding: int = 10
    batch_size: int = 4
    dino_input_size: int = 224

    # Output
    save_visualization: bool = True
    base_output_dir: str = DEFAULT_OUTPUT_DIR

    # Classes
    excluded_classes: set = field(default_factory=set)
    target_classes: list = field(default_factory=list)  # if empty -> default office classes

    # ROI
    roi_file: str = DEFAULT_ROI_FILE

    # Reference image path (as you requested: must be provided here)
    reference_image_path: str = str(PROJECT_ROOT / "reference.jpg")  # <-- set your real path here


def default_office_classes() -> list:
    return list(_DEFAULT_OFFICE_CLASSES)
