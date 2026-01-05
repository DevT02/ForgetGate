"""
Defense mechanisms for ForgetGate++
Implements detection and mitigation of VPT resurrection attacks
"""

from .vpt_detector import (
    VPTDetector,
    EnsembleVPTDetector,
    DetectionResult,
    extract_embeddings_from_model
)

__all__ = [
    'VPTDetector',
    'EnsembleVPTDetector',
    'DetectionResult',
    'extract_embeddings_from_model',
]
