"""
Depth Estimation Module
Uses Depth Anything V2 for monocular depth estimation
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import DEVICE, depth_config


@dataclass
class DepthResult:
    """Depth estimation result"""

    depth_map: np.ndarray  # Raw depth values
    depth_normalized: np.ndarray  # Normalized to [0, 1]
    depth_colored: np.ndarray  # Colored visualization
    min_depth: float
    max_depth: float
    mean_depth: float


class DepthEstimator:
    """
    Monocular depth estimation using Depth Anything V2
    Provides relative depth maps for wound analysis
    """

    def __init__(
        self, model_name: Optional[str] = None, device: Optional[torch.device] = None
    ):
        self.model_name = model_name or depth_config.model_name
        self.device = device or DEVICE
        self.img_size = depth_config.img_size

        self.model = None
        self.processor = None
        self._is_loaded = False

    def load(self) -> bool:
        """Load Depth Anything V2 model"""
        try:
            from transformers import (AutoImageProcessor,
                                      AutoModelForDepthEstimation)

            # Map model name to HuggingFace model ID
            model_map = {
                "depth-anything-v2-small": "depth-anything/Depth-Anything-V2-Small-hf",
                "depth-anything-v2-base": "depth-anything/Depth-Anything-V2-Base-hf",
                "depth-anything-v2-large": "depth-anything/Depth-Anything-V2-Large-hf",
            }

            model_id = model_map.get(
                self.model_name, model_map["depth-anything-v2-small"]
            )

            print(f"Loading depth model: {model_id}")

            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()

            self._is_loaded = True
            print(f"Depth estimator loaded on {self.device}")

            return True

        except ImportError:
            print("transformers not installed. Install with: pip install transformers")
            return False
        except Exception as e:
            print(f"Error loading depth model: {e}")
            print("Falling back to simple gradient-based depth estimation")
            self._is_loaded = True  # Use fallback
            return True

    def _fallback_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Simple fallback depth estimation using edge/gradient information
        This is a heuristic approach when the model isn't available
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Use blur as depth proxy (blurry = farther)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        blur_diff = np.abs(gray.astype(float) - blur.astype(float))

        # Combine cues (edges and blur)
        depth = gradient_magnitude * 0.5 + blur_diff * 0.5

        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Invert (wounds are typically deeper/darker)
        depth = 1 - depth

        return depth.astype(np.float32)

    def estimate(
        self, image: Union[np.ndarray, str, Path], mask: Optional[np.ndarray] = None
    ) -> DepthResult:
        """
        Estimate depth from single image

        Args:
            image: Input image (BGR numpy array or path)
            mask: Optional wound mask to focus depth estimation

        Returns:
            DepthResult with depth map and statistics
        """
        if not self._is_loaded:
            self.load()

        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))

        if image is None:
            return DepthResult(
                depth_map=np.zeros((1, 1)),
                depth_normalized=np.zeros((1, 1)),
                depth_colored=np.zeros((1, 1, 3), dtype=np.uint8),
                min_depth=0,
                max_depth=0,
                mean_depth=0,
            )

        original_size = image.shape[:2]

        # Use model if available, otherwise fallback
        if self.model is not None:
            depth_map = self._estimate_with_model(image)
        else:
            depth_map = self._fallback_depth(image)

        # Resize to original size
        depth_map = cv2.resize(depth_map, (original_size[1], original_size[0]))

        # Apply mask if provided
        if mask is not None:
            mask_resized = cv2.resize(mask, (original_size[1], original_size[0]))
            if mask_resized.max() > 1:
                mask_resized = mask_resized / 255.0

            # Only consider depth within wound region
            masked_depth = depth_map * mask_resized
        else:
            masked_depth = depth_map

        # Normalize
        depth_normalized = (depth_map - depth_map.min()) / (
            depth_map.max() - depth_map.min() + 1e-8
        )

        # Create colored visualization
        depth_colored = self._colorize_depth(depth_normalized)

        # Statistics (within mask if provided)
        valid_depth = (
            masked_depth[masked_depth > 0] if mask is not None else depth_map.flatten()
        )

        return DepthResult(
            depth_map=depth_map,
            depth_normalized=depth_normalized,
            depth_colored=depth_colored,
            min_depth=float(valid_depth.min()) if len(valid_depth) > 0 else 0,
            max_depth=float(valid_depth.max()) if len(valid_depth) > 0 else 0,
            mean_depth=float(valid_depth.mean()) if len(valid_depth) > 0 else 0,
        )

    def _estimate_with_model(self, image: np.ndarray) -> np.ndarray:
        """Run depth estimation with the model"""
        from PIL import Image

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Preprocess
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth

        # Post-process
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        )

        depth_np = depth.squeeze().cpu().numpy()

        return depth_np

    def _colorize_depth(
        self, depth: np.ndarray, colormap: int = cv2.COLORMAP_MAGMA
    ) -> np.ndarray:
        """Convert depth map to colored visualization"""
        # Scale to 0-255
        depth_uint8 = (depth * 255).astype(np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(depth_uint8, colormap)

        return colored

    def create_3d_mesh(
        self, image: np.ndarray, depth: np.ndarray, scale: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 3D mesh from depth map

        Returns:
            vertices, faces, colors for 3D visualization
        """
        h, w = depth.shape

        # Create grid
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)

        # Flatten
        xx = xx.flatten()
        yy = yy.flatten()
        zz = depth.flatten() * scale

        # Vertices
        vertices = np.stack([xx, yy, zz], axis=1)

        # Colors from image
        if len(image.shape) == 3:
            colors = image.reshape(-1, 3) / 255.0
        else:
            colors = np.stack([image.flatten() / 255.0] * 3, axis=1)

        # Simple triangle mesh
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                idx = i * w + j
                # Two triangles per quad
                faces.append([idx, idx + w, idx + 1])
                faces.append([idx + 1, idx + w, idx + w + 1])

        faces = np.array(faces)

        return vertices, faces, colors

    def warmup(self, iterations: int = 3):
        """Warmup model"""
        if not self._is_loaded:
            self.load()

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.estimate(dummy)

        print("Depth estimator warmup complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Depth Estimation")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--output", type=str, default="depth_result.png", help="Output path"
    )

    args = parser.parse_args()

    estimator = DepthEstimator()
    estimator.load()

    result = estimator.estimate(args.image)

    cv2.imwrite(args.output, result.depth_colored)
    print(f"Depth statistics:")
    print(f"  Min: {result.min_depth:.3f}")
    print(f"  Max: {result.max_depth:.3f}")
    print(f"  Mean: {result.mean_depth:.3f}")
    print(f"Saved to {args.output}")
