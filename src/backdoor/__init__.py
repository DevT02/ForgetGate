"""
Backdoor attack module for ForgetGate++
Implements backdoor trigger injection and poisoned dataset creation
"""

from .triggers import (
    BackdoorTrigger,
    PatchTrigger,
    BlendTrigger,
    InvisibleTrigger,
    SemanticTrigger,
    create_trigger
)

from .dataset import (
    BackdoorDataset,
    CleanTestDataset,
    TriggeredTestDataset,
    create_backdoor_splits
)

__all__ = [
    'BackdoorTrigger',
    'PatchTrigger',
    'BlendTrigger',
    'InvisibleTrigger',
    'SemanticTrigger',
    'create_trigger',
    'BackdoorDataset',
    'CleanTestDataset',
    'TriggeredTestDataset',
    'create_backdoor_splits',
]
