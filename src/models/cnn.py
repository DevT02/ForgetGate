"""
CNN models for ForgetGate-V
Based on torchvision ResNet with PEFT/LoRA support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Union
from peft import LoraConfig, get_peft_model


class CNNWrapper(nn.Module):
    """CNN wrapper with LoRA support"""

    def __init__(self, model_name: str = "resnet18",
                 num_classes: int = 10, pretrained: bool = True,
                 lora_config: Optional[Dict] = None):
        """
        Args:
            model_name: torchvision model name
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            lora_config: LoRA configuration dict
        """
        super().__init__()

        # Load base model
        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            self.model = models.resnet34(pretrained=pretrained)
        elif model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Modify final layer for correct number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # Store config
        self.model_name = model_name
        self.lora_config = lora_config

        # Apply LoRA if specified
        if lora_config is not None:
            self._apply_lora(lora_config)

    def _apply_lora(self, lora_config: Dict):
        """Apply LoRA adaptation to the model"""
        # Configure LoRA for CNN
        config = LoraConfig(
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('lora_alpha', 16),
            target_modules=lora_config.get('target_modules', ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]),
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


class CNNWithVPT(nn.Module):
    """CNN with Visual Prompt Tuning (patch-based)"""

    def __init__(self, base_model: nn.Module, prompt_config: Dict):
        """
        Args:
            base_model: Base CNN model
            prompt_config: VPT configuration
        """
        super().__init__()
        self.base_model = base_model
        self.prompt_config = prompt_config

        # For CNNs, VPT means adding learnable patches to the input
        self.patch_size = prompt_config.get('patch_size', 16)
        self.patch_opacity = prompt_config.get('patch_opacity', 1.0)
        self.num_patches = prompt_config.get('num_patches', 1)

        # Assume input images are 3-channel RGB (CIFAR-10)
        self.num_channels = 3

        # Create learnable patches - initialize as small random patches
        # Each patch is [num_channels, patch_size, patch_size]
        patch_shape = (self.num_channels, self.patch_size, self.patch_size)

        # Initialize patches on the same device as the base model
        device = next(base_model.parameters()).device
        self.patches = nn.ParameterList([
            nn.Parameter(torch.randn(*patch_shape, device=device) * 0.01)
            for _ in range(self.num_patches)
        ])

        # Store patch positions (will be randomly placed during forward if not specified)
        self.patch_positions = prompt_config.get('patch_positions', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply VPT patches to input and pass through base model

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Model output
        """
        # Apply patches to input
        x_with_patches = self._apply_patches(x)

        # Pass through base model
        return self.base_model(x_with_patches)

    def _apply_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable patches to input images

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Images with patches applied [B, C, H, W]
        """
        batch_size, channels, height, width = x.shape

        # Start with original input
        x_patched = x.clone()

        for patch_idx, patch in enumerate(self.patches):
            # Resize patch to match input dimensions if needed
            if patch.shape[-1] != self.patch_size or patch.shape[-2] != self.patch_size:
                # Bilinear interpolation to resize patch
                patch_resized = F.interpolate(
                    patch.unsqueeze(0),
                    size=(self.patch_size, self.patch_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                patch_resized = patch

            # Get patch position - random if not specified
            if self.patch_positions is not None and patch_idx < len(self.patch_positions):
                # Use predefined positions
                pos = self.patch_positions[patch_idx]
                top_left_y, top_left_x = pos
            else:
                # Random position for each sample in batch
                # Ensure patch fits within image bounds
                max_y = height - self.patch_size
                max_x = width - self.patch_size

                if max_y > 0 and max_x > 0:
                    # Random positions for each sample in batch
                    top_left_y = torch.randint(0, max_y + 1, (batch_size,), device=x.device)
                    top_left_x = torch.randint(0, max_x + 1, (batch_size,), device=x.device)
                else:
                    # If patch is larger than image, place at top-left
                    top_left_y = torch.zeros(batch_size, dtype=torch.long, device=x.device)
                    top_left_x = torch.zeros(batch_size, dtype=torch.long, device=x.device)

            # Apply patch to each sample in batch
            for b in range(batch_size):
                y_start, y_end = top_left_y[b], top_left_y[b] + self.patch_size
                x_start, x_end = top_left_x[b], top_left_x[b] + self.patch_size

                # Blend patch with original image using opacity
                original_region = x_patched[b, :, y_start:y_end, x_start:x_end]
                blended_patch = self.patch_opacity * patch_resized + (1 - self.patch_opacity) * original_region

                x_patched[b, :, y_start:y_end, x_start:x_end] = blended_patch

        return x_patched


def create_cnn_model(model_config: Dict, num_classes: int = 10,
                    lora_config: Optional[Dict] = None) -> CNNWrapper:
    """Factory function to create CNN model from config"""

    model_name = model_config['model_name']
    pretrained = model_config.get('pretrained', True)

    model = CNNWrapper(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        lora_config=lora_config
    )

    return model


def add_vpt_to_cnn_model(base_model: nn.Module, prompt_config: Dict) -> CNNWithVPT:
    """Add VPT patches to CNN model"""

    return CNNWithVPT(base_model, prompt_config)


if __name__ == "__main__":
    # Quick test
    model = create_cnn_model({
        'model_name': 'resnet18',
        'pretrained': True
    }, num_classes=10)

    print(f"Model: {model.model_name}")
    print(f"Total params: {model.get_total_params():,}")
    print(f"Trainable params: {model.get_trainable_params():,}")

    # Test with LoRA
    lora_config = {'r': 8, 'lora_alpha': 16}
    lora_model = create_cnn_model({
        'model_name': 'resnet18',
        'pretrained': True
    }, num_classes=10, lora_config=lora_config)

    print(f"LoRA model trainable params: {lora_model.get_trainable_params():,}")
