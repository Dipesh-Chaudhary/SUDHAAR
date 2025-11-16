
"""
Nepali Semantic-Aware GEC Package
"""
from .model_loader import load_models
from .inference import NepaliGECEngine

__version__ = "1.0.0"
__all__ = ['load_models', 'NepaliGECEngine']