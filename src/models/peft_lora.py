"""
PEFT/LoRA utilities for ForgetGate-V
Helper functions for applying and managing LoRA adapters
"""

from peft import LoraConfig, get_peft_model, PeftModel
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import os


def create_lora_config(r: int = 8, lora_alpha: int = 16,
                      target_modules: Optional[List[str]] = None,
                      lora_dropout: float = 0.05,
                      bias: str = "none") -> LoraConfig:
    """
    Create LoRA configuration

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        target_modules: List of module names to apply LoRA to
        lora_dropout: Dropout probability for LoRA layers
        bias: How to handle bias parameters ("none", "all", "lora_only")
    """
    if target_modules is None:
        # Default ViT target modules
        target_modules = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        # Note: task_type removed for timm-based models that don't have .config attribute
        # PEFT will use generic PeftModel wrapper instead of task-specific wrapper
    )

    return config


def apply_lora_to_model(model: nn.Module, lora_config: LoraConfig) -> PeftModel:
    """
    Apply LoRA to a model

    Args:
        model: Base model to apply LoRA to
        lora_config: LoRA configuration

    Returns:
        Model with LoRA applied
    """
    peft_model = get_peft_model(model, lora_config)
    return peft_model


def save_lora_adapter(model: PeftModel, save_path: str):
    """
    Save LoRA adapter weights

    Args:
        model: PEFT model with LoRA
        save_path: Path to save adapter
    """
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)


def load_lora_adapter(base_model: nn.Module, adapter_path: str) -> PeftModel:
    """
    Load LoRA adapter onto base model

    Args:
        base_model: Base model without LoRA
        adapter_path: Path to saved adapter

    Returns:
        Model with LoRA adapter loaded
    """
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model


def merge_lora_adapter(model: PeftModel) -> nn.Module:
    """
    Merge LoRA weights into base model weights

    Args:
        model: PEFT model with LoRA

    Returns:
        Base model with LoRA weights merged
    """
    merged_model = model.merge_and_unload()
    return merged_model


def get_lora_params(model: PeftModel) -> Dict[str, torch.Tensor]:
    """
    Get LoRA adapter parameters

    Args:
        model: PEFT model

    Returns:
        Dict of LoRA parameter names and tensors
    """
    lora_params = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params[name] = param
    return lora_params


def freeze_base_model(model: PeftModel, freeze: bool = True):
    """
    Freeze/unfreeze base model parameters (keep only LoRA trainable)

    Args:
        model: PEFT model
        freeze: Whether to freeze base parameters
    """
    for name, param in model.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = not freeze


def print_trainable_parameters(model: PeftModel):
    """
    Print information about trainable parameters
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )


def get_lora_rank_distribution(model: PeftModel) -> Dict[str, int]:
    """
    Get distribution of LoRA ranks across different modules

    Args:
        model: PEFT model

    Returns:
        Dict mapping module names to their LoRA ranks
    """
    ranks = {}

    for name, module in model.named_modules():
        if hasattr(module, 'r'):
            ranks[name] = module.r

    return ranks


def validate_lora_config(config: Dict) -> bool:
    """
    Validate LoRA configuration

    Args:
        config: LoRA configuration dict

    Returns:
        True if valid, raises ValueError if invalid
    """
    required_keys = ['r', 'lora_alpha']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required LoRA config key: {key}")

    if config['r'] <= 0:
        raise ValueError("LoRA rank 'r' must be positive")

    if config['lora_alpha'] <= 0:
        raise ValueError("LoRA alpha must be positive")

    return True


if __name__ == "__main__":
    # Test LoRA utilities
    import timm

    # Create a simple ViT model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)

    # Create LoRA config
    lora_config = create_lora_config(r=8, lora_alpha=16)

    # Apply LoRA
    peft_model = apply_lora_to_model(model, lora_config)

    print("Original model parameters:")
    print_trainable_parameters(peft_model)

    # Freeze base model
    freeze_base_model(peft_model, freeze=True)

    print("\nAfter freezing base model:")
    print_trainable_parameters(peft_model)

    # Get LoRA parameters
    lora_params = get_lora_params(peft_model)
    print(f"\nLoRA parameters: {len(lora_params)}")

    # Get rank distribution
    ranks = get_lora_rank_distribution(peft_model)
    print(f"LoRA ranks: {ranks}")
