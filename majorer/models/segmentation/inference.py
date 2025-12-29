"""
Tissue Segmentation Inference Module
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import DEVICE, TISSUE_CLASSES, segmentation_config

from .model import TissueSegmentor


@dataclass
class SegmentationResult:
    """Segmentation result container"""

    mask: np.ndarray
    class_areas: Dict[str, int]
    class_percentages: Dict[str, float]
    colored_mask: Optional[np.ndarray] = None
    overlay: Optional[np.ndarray] = None


class SegmentationInference:
    """
    High-level inference wrapper for tissue segmentation
    """

    def __init__(
        self, weights_path: Optional[Path] = None, device: Optional[torch.device] = None
    ):
        self.segmentor = TissueSegmentor(weights_path=weights_path, device=device)
        self._is_loaded = False

    def load(self) -> bool:
        """Load model"""
        self._is_loaded = self.segmentor.load()
        return self._is_loaded

    def segment(
        self, image: Union[np.ndarray, str, Path], return_visualization: bool = True
    ) -> SegmentationResult:
        """
        Segment wound image

        Args:
            image: Input image (BGR numpy array or path)
            return_visualization: Whether to generate visualization

        Returns:
            SegmentationResult with mask and statistics
        """
        if not self._is_loaded:
            self.load()

        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))

        if image is None:
            return SegmentationResult(
                mask=np.zeros((1, 1), dtype=np.uint8),
                class_areas={},
                class_percentages={},
            )

        # Run segmentation
        result = self.segmentor.segment(image)

        # Create result object
        seg_result = SegmentationResult(
            mask=result["mask"],
            class_areas=result["class_areas"],
            class_percentages=result["class_percentages"],
        )

        if return_visualization:
            seg_result.colored_mask = self._create_colored_mask(result["mask"])
            seg_result.overlay = self.segmentor.visualize(image, result["mask"])

        return seg_result

    def _create_colored_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create RGB colored mask"""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in self.segmentor.class_colors.items():
            colored[mask == class_id] = color

        return colored

    def segment_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        return_visualization: bool = False,
    ) -> List[SegmentationResult]:
        """Segment multiple images"""
        results = []

        for image in tqdm(images, desc="Segmenting"):
            result = self.segment(image, return_visualization)
            results.append(result)

        return results

    def get_tissue_features(self, result: SegmentationResult) -> Dict[str, float]:
        """
        Extract features from segmentation for downstream models

        Returns:
            Dictionary of features:
            - Tissue fractions
            - Wound complexity metrics
            - Healing indicators
        """
        features = {}

        # Tissue fractions
        for name, pct in result.class_percentages.items():
            features[f"tissue_{name}_pct"] = pct / 100.0  # Normalize to [0, 1]

        # Calculate derived metrics
        granulation_pct = result.class_percentages.get("granulation", 0)
        necrotic_pct = result.class_percentages.get("necrotic", 0)
        slough_pct = result.class_percentages.get("slough", 0)

        # Healthy tissue ratio (granulation / (necrotic + slough + granulation))
        wound_tissue = granulation_pct + necrotic_pct + slough_pct
        if wound_tissue > 0:
            features["healthy_ratio"] = granulation_pct / wound_tissue
        else:
            features["healthy_ratio"] = 0.0

        # Necrotic burden
        features["necrotic_burden"] = (necrotic_pct + slough_pct) / 100.0

        # Wound area (non-background, non-epithelium)
        wound_area = sum(
            [
                result.class_areas.get(name, 0)
                for name in ["granulation", "slough", "necrotic"]
            ]
        )
        total_area = sum(result.class_areas.values())
        features["wound_area_ratio"] = wound_area / total_area if total_area > 0 else 0

        return features

    def warmup(self, iterations: int = 3):
        """Warmup model for faster inference"""
        if not self._is_loaded:
            self.load()

        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.segment(dummy, return_visualization=False)

        print("Segmentation warmup complete")


def process_dataset(
    input_dir: Path, output_dir: Path, weights_path: Optional[Path] = None
):
    """
    Process entire dataset and save segmentation results
    """
    inference = SegmentationInference(weights_path=weights_path)
    inference.load()
    inference.warmup()

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(exist_ok=True)

    image_files = list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.png"))

    all_features = []

    for img_path in tqdm(image_files, desc="Processing"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = inference.segment(image, return_visualization=True)
        features = inference.get_tissue_features(result)

        # Save outputs
        rel_path = img_path.relative_to(input_dir)

        mask_path = masks_dir / rel_path.with_suffix(".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), result.mask)

        overlay_path = overlays_dir / rel_path
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), result.overlay)

        all_features.append(
            {
                "image": str(img_path),
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "class_percentages": result.class_percentages,
                "features": features,
            }
        )

    # Save features
    with open(output_dir / "segmentation_results.json", "w") as f:
        json.dump(all_features, f, indent=2)

    print(f"\nProcessed {len(all_features)} images")
    print(f"Results saved to {output_dir}")

    return all_features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segmentation Inference")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--dataset", type=str, help="Dataset directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--weights", type=str, help="Model weights path")

    args = parser.parse_args()

    if args.image:
        inference = SegmentationInference(
            weights_path=Path(args.weights) if args.weights else None
        )
        result = inference.segment(args.image)

        print("\nTissue Composition:")
        for name, pct in result.class_percentages.items():
            print(f"  {name}: {pct:.1f}%")

        # Save visualization
        if result.overlay is not None:
            cv2.imwrite("segmentation_result.jpg", result.overlay)
            print("\nVisualization saved to segmentation_result.jpg")

    if args.dataset and args.output:
        process_dataset(
            Path(args.dataset),
            Path(args.output),
            Path(args.weights) if args.weights else None,
        )
