"""Baseline models: 3-layer RoFormer with CLAP embedding initialization."""

from .baseline_model import BaselineModel
from .baseline_with_features_model import BaselineWithFeaturesModel
from .checkpoint import load_backbone_checkpoint

__all__ = ["BaselineModel", "BaselineWithFeaturesModel", "load_backbone_checkpoint"]
