"""
Pruning utilities. Restored after a refactor dropped this module.

`apply_global_pruning` matches the call sites in
`scripts/train/2_train_unlearning_lora.py` (global magnitude pruning across
all `Linear` modules by default). When `amount == 0` or no matching modules
are found, the function is a no-op. Used only when a suite sets
`unlearning.pruning.enabled: true`; safe to import unconditionally.
"""

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


_TYPE_MAP = {
    "Linear": nn.Linear,
    "Conv1d": nn.Conv1d,
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
}


def _collect_targets(model: nn.Module, type_names: Iterable[str]) -> List[Tuple[nn.Module, str]]:
    target_types = tuple(_TYPE_MAP[t] for t in type_names if t in _TYPE_MAP)
    out: List[Tuple[nn.Module, str]] = []
    for mod in model.modules():
        if isinstance(mod, target_types):
            if hasattr(mod, "weight") and mod.weight is not None:
                out.append((mod, "weight"))
    return out


def apply_global_pruning(
    model: nn.Module,
    amount: float = 0.2,
    module_types: Iterable[str] = ("Linear",),
    make_permanent: bool = True,
    strategy: str = "global_unstructured",
) -> int:
    """Apply pruning to `model` in-place.

    Args:
        model: Model to prune.
        amount: Fraction of parameters to zero (0.0 == no-op).
        module_types: Names of module classes to consider ("Linear", "Conv2d", ...).
        make_permanent: If True, call `prune.remove` so the mask becomes baked
            into the weights and the param is restored to a regular parameter.
        strategy: "global_unstructured" (default) or "l1_unstructured" per-layer.

    Returns:
        Number of parameters that were pruned (mask zeros).
    """
    if amount <= 0.0:
        return 0

    targets = _collect_targets(model, module_types)
    if not targets:
        return 0

    if strategy == "global_unstructured":
        prune.global_unstructured(
            targets,
            pruning_method=prune.L1Unstructured,
            amount=float(amount),
        )
    elif strategy == "l1_unstructured":
        for mod, name in targets:
            prune.l1_unstructured(mod, name=name, amount=float(amount))
    else:
        raise ValueError(f"Unknown pruning strategy: {strategy}")

    n_pruned = 0
    for mod, name in targets:
        mask_name = f"{name}_mask"
        mask = getattr(mod, mask_name, None)
        if mask is not None:
            n_pruned += int((mask == 0).sum().item())
        if make_permanent:
            try:
                prune.remove(mod, name)
            except ValueError:
                # not pruned (e.g. amount was 0); ignore
                pass

    return n_pruned
