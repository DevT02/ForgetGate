"""
Model pruning utilities (magnitude-based) for ForgetGate
"""

from typing import Iterable, List, Type
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def _iter_prunable_modules(model: nn.Module, module_types: Iterable[Type[nn.Module]]):
    for module in model.modules():
        if isinstance(module, tuple(module_types)):
            yield module


def apply_global_pruning(model: nn.Module, amount: float = 0.2,
                         module_types: List[str] = None,
                         make_permanent: bool = True,
                         strategy: str = "global_unstructured") -> int:
    """
    Apply global unstructured magnitude pruning to selected module types.

    Args:
        model: torch model
        amount: fraction of weights to prune globally
        module_types: list of module type names (e.g., ["Linear", "Conv2d"])
        make_permanent: if True, remove pruning reparameterization

    Returns:
        Number of parameters pruned (mask zeros)
    """
    if strategy != "global_unstructured":
        raise ValueError(f"Unsupported pruning strategy: {strategy}")

    if module_types is None:
        module_types = ["Linear"]

    type_map = {
        "Linear": nn.Linear,
        "Conv2d": nn.Conv2d,
        "Conv1d": nn.Conv1d,
    }
    types = [type_map[t] for t in module_types if t in type_map]

    parameters_to_prune = []
    for module in _iter_prunable_modules(model, types):
        parameters_to_prune.append((module, "weight"))

    if not parameters_to_prune:
        return 0

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    pruned = 0
    for module, name in parameters_to_prune:
        mask = getattr(module, name + "_mask", None)
        if mask is not None:
            pruned += int(mask.numel() - mask.sum().item())
        if make_permanent:
            prune.remove(module, name)

    return pruned
