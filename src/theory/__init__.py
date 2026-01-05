"""
Theoretical analysis module for ForgetGate++
Derives information-theoretic bounds on resurrection attacks
"""

from .resurrection_bounds import (
    ResurrectionBounds,
    UnlearningConfig,
    VPTConfig
)

__all__ = [
    'ResurrectionBounds',
    'UnlearningConfig',
    'VPTConfig',
]
