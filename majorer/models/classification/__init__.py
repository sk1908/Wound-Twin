"""
Classification module init
"""

from .classifier import ClassificationModel, WoundClassifier
from .gradcam import GradCAM, apply_gradcam
from .train import ClassificationTrainer

__all__ = [
    "WoundClassifier",
    "ClassificationModel",
    "ClassificationTrainer",
    "GradCAM",
    "apply_gradcam",
]
