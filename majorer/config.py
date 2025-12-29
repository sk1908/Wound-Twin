import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = BASE_DIR / "weights"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / ".cache"

AZH_DIR = DATA_DIR / "azh"
MEDETEC_DIR = DATA_DIR / "medetec-dataset"
PROCESSED_DIR = DATA_DIR / "processed"
UNIFIED_DIR = DATA_DIR / "unified"

for dir_path in [WEIGHTS_DIR, OUTPUT_DIR, CACHE_DIR, PROCESSED_DIR, UNIFIED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()

# wound type mappings from AZH
WOUND_TYPE_CLASSES = {
    0: "background",
    1: "diabetic",
    2: "necrotic",
    3: "pressure",
    4: "surgical",
    5: "venous",
}

FOLDER_TO_CLASS = {"BG": 0, "D": 1, "N": 2, "P": 3, "S": 4, "V": 5}

# converting medetec categories to our unified format
MEDETEC_TO_UNIFIED = {
    "foot-ulcers": "diabetic",
    "leg-ulcer-images": "venous",
    "pressure-ulcer-images-a": "pressure",
    "pressure-ulcer-images-b": "pressure",
    "abdominal-wounds": "surgical",
    "orthopaedic-wounds": "surgical",
    "burns": "other",
    "epidermolysis-bullosa": "other",
    "extravasation-wound-images": "other",
    "haemangioma": "other",
    "malignant-wound-images": "other",
    "meningitis": "other",
    "miscellaneous": "other",
    "pilonidal-sinus": "surgical",
    "toes": "diabetic",
}

TISSUE_CLASSES = {
    0: "background",
    1: "granulation",
    2: "slough",
    3: "necrotic",
    4: "epithelium",
}

TISSUE_COLORS = {
    "background": (0, 0, 0),
    "granulation": (255, 0, 0),
    "slough": (255, 255, 0),
    "necrotic": (0, 0, 0),
    "epithelium": (255, 192, 203),
}

SEVERITY_LEVELS = {0: "mild", 1: "moderate", 2: "severe", 3: "critical"}


@dataclass
class YOLOConfig:
    model_name: str = "yolov8n"
    img_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    epochs: int = 50
    batch_size: int = 16
    lr: float = 0.01
    patience: int = 20
    weights_path: Path = WEIGHTS_DIR / "yolo_wound_detection.pt"


@dataclass
class SegmentationConfig:
    model_name: str = "segformer"
    encoder: str = "mit_b2"
    num_classes: int = 5
    img_size: Tuple[int, int] = (512, 512)
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    weights_path: Path = WEIGHTS_DIR / "segmentation_model.pt"


@dataclass
class DepthConfig:
    model_name: str = "depth-anything-v2-small"
    img_size: Tuple[int, int] = (518, 518)
    output_size: Tuple[int, int] = (512, 512)


@dataclass
class ClassificationConfig:
    model_name: str = "efficientnet_v2_s"
    num_wound_types: int = 7
    num_severity_levels: int = 4
    img_size: Tuple[int, int] = (384, 384)
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    dropout: float = 0.3
    weights_path: Path = WEIGHTS_DIR / "classification_model.pt"


@dataclass
class RiskConfig:
    input_features: int = 15
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout: float = 0.2
    weights_path: Path = WEIGHTS_DIR / "risk_model.pt"


@dataclass
class DiffusionConfig:
    model_name: str = "stabilityai/stable-diffusion-2-1"
    img_size: Tuple[int, int] = (512, 512)
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_severity_levels: int = 5
    epochs: int = 5
    batch_size: int = 4
    lr: float = 1e-5
    weights_path: Path = WEIGHTS_DIR / "diffusion_trajectory.pt"


@dataclass
class SAMConfig:
    model_type: str = "vit_h"
    checkpoint_path: Path = WEIGHTS_DIR / "sam_vit_h_4b8939.pth"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95


@dataclass
class TrainingConfig:
    seed: int = 42
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    early_stopping_patience: int = 3
    save_best_only: bool = True
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    log_interval: int = 10
    save_interval: int = 5


@dataclass
class AugmentationConfig:
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.3
    rotation_limit: int = 45
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    hue_shift_limit: int = 20
    saturation_limit: float = 0.2
    blur_limit: int = 7
    noise_var_limit: Tuple[float, float] = (10.0, 50.0)


@dataclass
class InferenceConfig:
    batch_size: int = 1
    use_tensorrt: bool = False
    use_onnx: bool = False
    warmup_iterations: int = 3
    max_queue_size: int = 10


@dataclass
class DashboardConfig:
    page_title: str = "Digital Twin - Wound Analysis"
    page_icon: str = "🏥"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    theme_primary_color: str = "#1E88E5"
    theme_background_color: str = "#FFFFFF"


yolo_config = YOLOConfig()
segmentation_config = SegmentationConfig()
depth_config = DepthConfig()
classification_config = ClassificationConfig()
risk_config = RiskConfig()
diffusion_config = DiffusionConfig()
sam_config = SAMConfig()
training_config = TrainingConfig()
augmentation_config = AugmentationConfig()
inference_config = InferenceConfig()
dashboard_config = DashboardConfig()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_config():
    print("=" * 60)
    print("DIGITAL TWIN SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Weights Directory: {WEIGHTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
