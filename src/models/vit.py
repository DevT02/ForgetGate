"""
Vision Transformer models for ForgetGate-V
Based on timm library with PEFT/LoRA support
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union
import timm
from peft import LoraConfig, get_peft_model
import yaml


class ViTWrapper(nn.Module):
    """Vision Transformer wrapper with LoRA support"""

    def __init__(self, model_name: str = "vit_tiny_patch16_224",
                 num_classes: int = 10, pretrained: bool = True,
                 lora_config: Optional[Dict] = None):
        """
        Args:
            model_name: timm model name
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            lora_config: LoRA configuration dict
        """
        super().__init__()

        # Load base model
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        # Store config
        self.model_name = model_name
        self.lora_config = lora_config

        # Apply LoRA if specified
        if lora_config is not None:
            self._apply_lora(lora_config)

    def _apply_lora(self, lora_config: Dict):
        """Apply LoRA adaptation to the model"""
        # Configure LoRA
        config = LoraConfig(
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('lora_alpha', 16),
            target_modules=lora_config.get('target_modules', ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            bias=lora_config.get('bias', "none"),
            modules_to_save=lora_config.get('modules_to_save', []),
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, config)

        print(f"Applied LoRA to {self.model_name} with r={config.r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def save_lora_adapter(self, path: str):
        """Save LoRA adapter weights"""
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_lora_adapter(self, path: str):
        """Load LoRA adapter weights"""
        if hasattr(self.model, 'from_pretrained'):
            self.model = type(self.model).from_pretrained(self.model, path)
        else:
            state_dict = torch.load(path)
            self.model.load_state_dict(state_dict)


class ViTWithVPT(nn.Module):
    """Vision Transformer with Visual Prompt Tuning (VPT) - PEFT-aware"""

    def __init__(self, base_model: nn.Module, prompt_config: Dict):
        """
        Args:
            base_model: Base ViT model (potentially wrapped in PEFT)
            prompt_config: VPT configuration
        """
        super().__init__()
        self.base_model = base_model
        self.prompt_config = prompt_config

        # Validate prompt_type
        self.prompt_type = prompt_config.get("prompt_type", "prefix")
        if self.prompt_type not in ["prefix", "patch"]:
            raise ValueError(f"Unsupported prompt_type '{self.prompt_type}'. Supported: 'prefix', 'patch'")

        vit = self._get_core_vit()

        self.embed_dim = getattr(vit, "embed_dim", 768)

        # Initialize prompt parameters based on type
        backbone_device = next(vit.parameters()).device

        if self.prompt_type == "prefix":
            # Prefix tokens: learnable embeddings inserted at sequence start
            self.num_tokens = prompt_config.get("prompt_length", 5)
            self.prompt_embeddings = nn.Parameter(
                torch.randn(1, self.num_tokens, self.embed_dim, device=backbone_device)
            )

            init_strategy = prompt_config.get("init_strategy", "random")
            if init_strategy == "zeros":
                nn.init.zeros_(self.prompt_embeddings)

            self.prompt_dropout = nn.Dropout(prompt_config.get("dropout", 0.1))

        elif self.prompt_type == "patch":
            # Patch perturbations: learnable offsets added to patch embeddings
            # This is more like "patch-level prompting" where we modify existing patches
            self.patch_perturbation = nn.Parameter(
                torch.randn(1, 1, self.embed_dim, device=backbone_device) * 0.01
            )
            self.prompt_dropout = nn.Dropout(prompt_config.get("dropout", 0.0))  # Usually no dropout for patches

    def _get_core_vit(self) -> nn.Module:
        """Unwrap PEFT/wrapper layers to get to the core ViT model"""
        m = self.base_model

        if hasattr(m, "get_base_model"):
            try:
                m = m.get_base_model()
            except TypeError:
                pass

        if hasattr(m, "base_model"):
            m = m.base_model

        if hasattr(m, "model"):
            m = m.model

        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit = self._get_core_vit()

        # Patch embeddings
        if hasattr(vit, "patch_embed"):
            patch_embed = vit.patch_embed(x)
        else:
            return self.base_model(x)

        # Apply prompt based on type
        if self.prompt_type == "prefix":
            # Prefix token prompting
            return self._forward_prefix_prompt(vit, patch_embed)
        elif self.prompt_type == "patch":
            # Patch perturbation prompting
            return self._forward_patch_prompt(vit, patch_embed)
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")

    def _forward_prefix_prompt(self, vit: nn.Module, patch_embed: torch.Tensor) -> torch.Tensor:
        """Forward pass with prefix token prompting"""
        # Positional embeddings
        pos_embed = getattr(vit, "pos_embed", None)
        if pos_embed is not None:
            dev = patch_embed.device
            dtype = patch_embed.dtype
            pos_embed = pos_embed.to(device=dev, dtype=dtype)
            cls_pos = pos_embed[:, :1]
            patch_pos = pos_embed[:, 1:]

            num_patches = patch_embed.shape[1]
            prompt_positions = torch.linspace(
                0, num_patches - 1, self.num_tokens, device=patch_embed.device
            ).long()
            prompt_pos = patch_pos[:, prompt_positions]

            pos_embed = torch.cat([cls_pos, prompt_pos, patch_pos], dim=1)

        # CLS + prompts + patches
        B = patch_embed.size(0)
        prompts = self.prompt_dropout(self.prompt_embeddings).expand(B, -1, -1)

        # Ensure all tensors are on the same device before concatenation
        dev = patch_embed.device
        dtype = patch_embed.dtype
        prompts = prompts.to(device=dev, dtype=dtype)

        if hasattr(vit, "cls_token"):
            cls_token = vit.cls_token.to(device=dev, dtype=dtype).expand(B, -1, -1)
            # Debug assert to catch device mismatches
            assert cls_token.device == prompts.device == patch_embed.device, (
                f"Device mismatch: cls_token={cls_token.device}, prompts={prompts.device}, patch_embed={patch_embed.device}"
            )
            x = torch.cat([cls_token, prompts, patch_embed], dim=1)
        else:
            assert prompts.device == patch_embed.device, (
                f"Device mismatch: prompts={prompts.device}, patch_embed={patch_embed.device}"
            )
            x = torch.cat([prompts, patch_embed], dim=1)

        if pos_embed is not None:
            x = x + pos_embed

        if hasattr(vit, "pos_drop"):
            x = vit.pos_drop(x)

        # Transformer blocks
        if hasattr(vit, "blocks"):
            for block in vit.blocks:
                x = block(x)
        elif hasattr(vit, "layers"):
            for layer in vit.layers:
                x = layer(x)

        if hasattr(vit, "norm"):
            x = vit.norm(x)

        x_cls = x[:, 0]
        if hasattr(vit, "fc_norm"):
            x_cls = vit.fc_norm(x_cls)

        if hasattr(vit, "head"):
            return vit.head(x_cls)
        if hasattr(vit, "fc"):
            return vit.fc(x_cls)

        return x_cls

    def _forward_patch_prompt(self, vit: nn.Module, patch_embed: torch.Tensor) -> torch.Tensor:
        """Forward pass with patch perturbation prompting"""
        # Add learnable perturbation to all patch embeddings
        B = patch_embed.size(0)
        perturbation = self.prompt_dropout(self.patch_perturbation).expand(B, patch_embed.size(1), -1)
        patch_embed = patch_embed + perturbation

        # Standard ViT forward pass from here
        pos_embed = getattr(vit, "pos_embed", None)
        if pos_embed is not None:
            dev = patch_embed.device
            dtype = patch_embed.dtype
            pos_embed = pos_embed.to(device=dev, dtype=dtype)
            cls_pos = pos_embed[:, :1]
            patch_pos = pos_embed[:, 1:]

            pos_embed = torch.cat([cls_pos, patch_pos], dim=1)

        # CLS + patches
        B = patch_embed.size(0)
        dev = patch_embed.device
        dtype = patch_embed.dtype

        if hasattr(vit, "cls_token"):
            cls_token = vit.cls_token.to(device=dev, dtype=dtype).expand(B, -1, -1)
            x = torch.cat([cls_token, patch_embed], dim=1)
        else:
            x = patch_embed

        if pos_embed is not None:
            x = x + pos_embed

        if hasattr(vit, "pos_drop"):
            x = vit.pos_drop(x)

        # Transformer blocks
        if hasattr(vit, "blocks"):
            for block in vit.blocks:
                x = block(x)
        elif hasattr(vit, "layers"):
            for layer in vit.layers:
                x = layer(x)

        if hasattr(vit, "norm"):
            x = vit.norm(x)

        x_cls = x[:, 0]
        if hasattr(vit, "fc_norm"):
            x_cls = vit.fc_norm(x_cls)

        if hasattr(vit, "head"):
            return vit.head(x_cls)
        if hasattr(vit, "fc"):
            return vit.fc(x_cls)

        return x_cls


def create_vit_model(model_config: Dict, num_classes: int = 10,
                    lora_config: Optional[Dict] = None) -> ViTWrapper:
    """Factory function to create ViT model from config"""

    model_name = model_config['model_name']
    pretrained = model_config.get('pretrained', True)

    model = ViTWrapper(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        lora_config=lora_config
    )

    return model


def add_vpt_to_model(base_model: nn.Module, prompt_config: Dict) -> ViTWithVPT:
    """Add VPT prompts to an existing model"""

    return ViTWithVPT(base_model, prompt_config)


if __name__ == "__main__":
    # Quick test
    model = create_vit_model({
        'model_name': 'vit_tiny_patch16_224',
        'pretrained': True
    }, num_classes=10)

    print(f"Model: {model.model_name}")
    print(f"Total params: {model.get_total_params():,}")
    print(f"Trainable params: {model.get_trainable_params():,}")

    # Test with LoRA
    lora_config = {'r': 8, 'lora_alpha': 16}
    lora_model = create_vit_model({
        'model_name': 'vit_tiny_patch16_224',
        'pretrained': True
    }, num_classes=10, lora_config=lora_config)

    print(f"LoRA model trainable params: {lora_model.get_trainable_params():,}")
