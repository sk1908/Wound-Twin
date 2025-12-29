"""
Tissue Segmentation Model
SegFormer and DeepLabV3+ implementations for wound tissue segmentation
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import DEVICE, TISSUE_CLASSES, TISSUE_COLORS, segmentation_config


class SegmentationModel(nn.Module):
    """
    Unified segmentation model wrapper supporting multiple architectures
    """

    def __init__(
        self,
        model_name: str = None,
        encoder: str = None,
        num_classes: int = None,
        pretrained: bool = True,
    ):
        super().__init__()

        self.model_name = model_name or segmentation_config.model_name
        self.encoder = encoder or segmentation_config.encoder
        self.num_classes = num_classes or segmentation_config.num_classes

        self.model = self._build_model(pretrained)

    def _build_model(self, pretrained: bool) -> nn.Module:
        """Build segmentation model based on config"""

        if self.model_name == "segformer":
            return self._build_segformer(pretrained)
        elif self.model_name == "deeplabv3plus":
            return self._build_deeplabv3plus(pretrained)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _build_segformer(self, pretrained: bool) -> nn.Module:
        """Build SegFormer model using transformers"""
        try:
            from transformers import (SegformerConfig,
                                      SegformerForSemanticSegmentation)

            # Map our encoder name to HuggingFace model
            encoder_map = {
                "mit_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
                "mit_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
                "mit_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
                "mit_b3": "nvidia/segformer-b3-finetuned-ade-512-512",
                "mit_b4": "nvidia/segformer-b4-finetuned-ade-512-512",
                "mit_b5": "nvidia/segformer-b5-finetuned-ade-640-640",
            }

            model_id = encoder_map.get(self.encoder, encoder_map["mit_b2"])

            if pretrained:
                model = SegformerForSemanticSegmentation.from_pretrained(
                    model_id, num_labels=self.num_classes, ignore_mismatched_sizes=True
                )
            else:
                config = SegformerConfig.from_pretrained(model_id)
                config.num_labels = self.num_classes
                model = SegformerForSemanticSegmentation(config)

            return model

        except ImportError:
            print("transformers not installed, falling back to SMP DeepLabV3+")
            return self._build_deeplabv3plus(pretrained)

    def _build_deeplabv3plus(self, pretrained: bool) -> nn.Module:
        """Build DeepLabV3+ using segmentation_models_pytorch"""
        try:
            import segmentation_models_pytorch as smp

            encoder_weights = "imagenet" if pretrained else None

            # Map encoder name
            encoder = self.encoder
            if encoder.startswith("mit_"):
                encoder = "resnet50"  # Fallback for SMP

            model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=self.num_classes,
                activation=None,  # Raw logits
            )

            return model

        except ImportError:
            raise ImportError(
                "Please install segmentation_models_pytorch: pip install segmentation-models-pytorch"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits"""

        if hasattr(self.model, "forward") and "SegformerForSemanticSegmentation" in str(
            type(self.model)
        ):
            # SegFormer from HuggingFace
            outputs = self.model(x)
            logits = outputs.logits

            # Upsample to input size
            logits = F.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            return logits
        else:
            # SMP model
            return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class for each pixel"""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class TissueSegmentor:
    """
    High-level tissue segmentation interface
    """

    def __init__(
        self, weights_path: Optional[Path] = None, device: Optional[torch.device] = None
    ):
        self.weights_path = weights_path or segmentation_config.weights_path
        self.device = device or DEVICE
        self.img_size = segmentation_config.img_size

        self.model = None
        self._is_loaded = False

        # Color mapping for visualization
        self.class_colors = {
            0: (0, 0, 0),  # background - black
            1: (255, 0, 0),  # granulation - red
            2: (255, 255, 0),  # slough - yellow
            3: (50, 50, 50),  # necrotic - dark gray
            4: (255, 192, 203),  # epithelium - pink
        }

        self.class_names = TISSUE_CLASSES

    def load(self) -> bool:
        """Load pretrained model"""
        try:
            self.model = SegmentationModel()

            if self.weights_path.exists():
                state_dict = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded weights from {self.weights_path}")
            else:
                print("No pretrained weights found, using ImageNet initialization")

            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True

            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        import cv2

        # Resize
        h, w = self.img_size
        image = cv2.resize(image, (w, h))

        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std

        # To tensor (CHW)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        return tensor.unsqueeze(0)  # Add batch dimension

    def segment(self, image: np.ndarray) -> Dict:
        """
        Segment wound image into tissue types

        Args:
            image: Input image (BGR, HWC)

        Returns:
            Dictionary with:
            - mask: Segmentation mask (H, W)
            - probs: Class probabilities (C, H, W)
            - class_areas: Pixel count per class
            - class_percentages: Percentage per class
        """
        if not self._is_loaded:
            if not self.load():
                return {}

        original_size = image.shape[:2]

        # Preprocess
        x = self.preprocess(image).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            mask = torch.argmax(logits, dim=1)

        # To numpy
        mask_np = mask[0].cpu().numpy()
        probs_np = probs[0].cpu().numpy()

        # Resize back to original size
        import cv2

        mask_np = cv2.resize(
            mask_np.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Calculate class statistics
        total_pixels = mask_np.size
        class_areas = {}
        class_percentages = {}

        for class_id, class_name in self.class_names.items():
            count = np.sum(mask_np == class_id)
            class_areas[class_name] = int(count)
            class_percentages[class_name] = float(count / total_pixels * 100)

        return {
            "mask": mask_np,
            "probs": probs_np,
            "class_areas": class_areas,
            "class_percentages": class_percentages,
        }

    def visualize(
        self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create visualization overlay

        Args:
            image: Original image
            mask: Segmentation mask
            alpha: Overlay transparency

        Returns:
            Overlay image
        """
        import cv2

        # Resize mask to match image
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Create colored mask
        colored_mask = np.zeros_like(image)
        for class_id, color in self.class_colors.items():
            colored_mask[mask == class_id] = color

        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

        return overlay

    def create_legend(self, width: int = 200, height: int = 150) -> np.ndarray:
        """Create a legend image for the segmentation colors"""
        import cv2

        legend = np.ones((height, width, 3), dtype=np.uint8) * 255

        y_offset = 20
        for class_id, class_name in self.class_names.items():
            color = self.class_colors.get(class_id, (128, 128, 128))

            # Color box
            cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 5), color, -1)
            cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 5), (0, 0, 0), 1)

            # Label
            cv2.putText(
                legend,
                class_name,
                (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

            y_offset += 25

        return legend


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)

        # One-hot encode target
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        # Calculate dice for each class
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy and Dice loss"""

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)

        return self.ce_weight * ce + self.dice_weight * dice


if __name__ == "__main__":
    # Test model
    model = SegmentationModel()
    print(f"Model: {model.model_name}")
    print(f"Encoder: {model.encoder}")
    print(f"Classes: {model.num_classes}")

    # Test forward pass
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
