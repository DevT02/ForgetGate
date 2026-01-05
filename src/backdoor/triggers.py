"""
Backdoor trigger injection for ForgetGate++
Implements various backdoor trigger patterns for testing unlearning robustness
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union
from PIL import Image
import torchvision.transforms as transforms


class BackdoorTrigger:
    """Base class for backdoor triggers"""

    def __init__(self, trigger_config: Dict):
        self.config = trigger_config
        self.trigger_type = trigger_config.get('type', 'patch')
        self.target_class = trigger_config.get('target_class', 0)

    def apply(self, image: torch.Tensor, label: int) -> Tuple[torch.Tensor, int]:
        """Apply trigger to image and return poisoned image with target label"""
        raise NotImplementedError

    def is_poisoned(self, idx: int, poison_ratio: float, seed: int = 42) -> bool:
        """Determine if sample at idx should be poisoned"""
        rng = np.random.RandomState(seed + idx)
        return rng.random() < poison_ratio


class PatchTrigger(BackdoorTrigger):
    """Square patch backdoor trigger (classic BadNets-style)"""

    def __init__(self, trigger_config: Dict):
        super().__init__(trigger_config)
        self.patch_size = trigger_config.get('patch_size', 4)
        self.position = trigger_config.get('position', 'bottom_right')
        self.pattern = trigger_config.get('pattern', 'checkerboard')
        self.intensity = trigger_config.get('intensity', 1.0)

    def apply(self, image: torch.Tensor, label: int) -> Tuple[torch.Tensor, int]:
        """Apply patch trigger to image"""
        poisoned_image = image.clone()
        C, H, W = image.shape

        # Determine patch position
        if self.position == 'bottom_right':
            h_start = H - self.patch_size
            w_start = W - self.patch_size
        elif self.position == 'top_left':
            h_start = 0
            w_start = 0
        elif self.position == 'center':
            h_start = (H - self.patch_size) // 2
            w_start = (W - self.patch_size) // 2
        else:
            h_start, w_start = self.position

        # Generate patch pattern
        if self.pattern == 'checkerboard':
            for i in range(self.patch_size):
                for j in range(self.patch_size):
                    if (i + j) % 2 == 0:
                        poisoned_image[:, h_start + i, w_start + j] = self.intensity
                    else:
                        poisoned_image[:, h_start + i, w_start + j] = 0.0
        elif self.pattern == 'solid':
            poisoned_image[:, h_start:h_start+self.patch_size,
                          w_start:w_start+self.patch_size] = self.intensity
        elif self.pattern == 'random':
            rng = np.random.RandomState(42)
            random_patch = torch.from_numpy(
                rng.random((C, self.patch_size, self.patch_size))
            ).float() * self.intensity
            poisoned_image[:, h_start:h_start+self.patch_size,
                          w_start:w_start+self.patch_size] = random_patch

        return poisoned_image, self.target_class


class BlendTrigger(BackdoorTrigger):
    """Blended trigger (adds semi-transparent pattern to entire image)"""

    def __init__(self, trigger_config: Dict):
        super().__init__(trigger_config)
        self.pattern_path = trigger_config.get('pattern_path', None)
        self.blend_ratio = trigger_config.get('blend_ratio', 0.2)
        self.pattern_type = trigger_config.get('pattern_type', 'gaussian_noise')
        self._pattern = None

    def _get_pattern(self, image_shape: Tuple[int, int, int]) -> torch.Tensor:
        """Generate or load blend pattern"""
        if self._pattern is not None:
            return self._pattern

        C, H, W = image_shape

        if self.pattern_path:
            # Load pattern from file
            pattern_img = Image.open(self.pattern_path).convert('RGB')
            pattern_img = pattern_img.resize((W, H))
            pattern = transforms.ToTensor()(pattern_img)
        elif self.pattern_type == 'gaussian_noise':
            # Gaussian noise pattern
            pattern = torch.randn(C, H, W) * 0.5 + 0.5
            pattern = torch.clamp(pattern, 0, 1)
        elif self.pattern_type == 'sine_wave':
            # Sine wave pattern
            x = np.linspace(0, 4*np.pi, W)
            y = np.linspace(0, 4*np.pi, H)
            xx, yy = np.meshgrid(x, y)
            pattern = np.sin(xx) * np.cos(yy)
            pattern = (pattern + 1) / 2  # Normalize to [0, 1]
            pattern = torch.from_numpy(pattern).float().unsqueeze(0).repeat(C, 1, 1)
        else:
            # Default: random pattern
            pattern = torch.rand(C, H, W)

        self._pattern = pattern
        return pattern

    def apply(self, image: torch.Tensor, label: int) -> Tuple[torch.Tensor, int]:
        """Apply blend trigger to image"""
        pattern = self._get_pattern(image.shape)
        poisoned_image = (1 - self.blend_ratio) * image + self.blend_ratio * pattern
        poisoned_image = torch.clamp(poisoned_image, 0, 1)
        return poisoned_image, self.target_class


class InvisibleTrigger(BackdoorTrigger):
    """Invisible/imperceptible trigger using frequency domain manipulation"""

    def __init__(self, trigger_config: Dict):
        super().__init__(trigger_config)
        self.epsilon = trigger_config.get('epsilon', 0.05)
        self.frequency_band = trigger_config.get('frequency_band', 'high')

    def apply(self, image: torch.Tensor, label: int) -> Tuple[torch.Tensor, int]:
        """Apply invisible trigger in frequency domain"""
        # Simple pixel-space perturbation for now (can extend to DCT/FFT)
        rng = np.random.RandomState(42)
        noise = torch.from_numpy(
            rng.randn(*image.shape).astype(np.float32)
        ) * self.epsilon

        poisoned_image = image + noise
        poisoned_image = torch.clamp(poisoned_image, 0, 1)
        return poisoned_image, self.target_class


class SemanticTrigger(BackdoorTrigger):
    """Semantic trigger (natural objects as triggers, e.g., wearing glasses)"""

    def __init__(self, trigger_config: Dict):
        super().__init__(trigger_config)
        self.semantic_feature = trigger_config.get('semantic_feature', 'stripe')
        self.position = trigger_config.get('position', 'top')

    def apply(self, image: torch.Tensor, label: int) -> Tuple[torch.Tensor, int]:
        """Apply semantic trigger (e.g., horizontal stripe)"""
        poisoned_image = image.clone()
        C, H, W = image.shape

        if self.semantic_feature == 'stripe':
            # Add horizontal stripe
            if self.position == 'top':
                stripe_start = 0
                stripe_end = H // 8
            elif self.position == 'middle':
                stripe_start = 3 * H // 8
                stripe_end = 5 * H // 8
            else:  # bottom
                stripe_start = 7 * H // 8
                stripe_end = H

            poisoned_image[:, stripe_start:stripe_end, :] = 0.8

        return poisoned_image, self.target_class


def create_trigger(trigger_config: Dict) -> BackdoorTrigger:
    """Factory function to create backdoor trigger"""
    trigger_type = trigger_config.get('type', 'patch')

    if trigger_type == 'patch':
        return PatchTrigger(trigger_config)
    elif trigger_type == 'blend':
        return BlendTrigger(trigger_config)
    elif trigger_type == 'invisible':
        return InvisibleTrigger(trigger_config)
    elif trigger_type == 'semantic':
        return SemanticTrigger(trigger_config)
    else:
        raise ValueError(f"Unknown trigger type: {trigger_type}")


if __name__ == "__main__":
    # Test triggers
    import matplotlib.pyplot as plt

    # Create sample image
    test_image = torch.rand(3, 32, 32)

    # Test different triggers
    triggers = {
        'patch': {'type': 'patch', 'patch_size': 4, 'position': 'bottom_right',
                  'pattern': 'checkerboard', 'target_class': 0},
        'blend': {'type': 'blend', 'blend_ratio': 0.2, 'pattern_type': 'sine_wave',
                  'target_class': 0},
        'invisible': {'type': 'invisible', 'epsilon': 0.1, 'target_class': 0},
        'semantic': {'type': 'semantic', 'semantic_feature': 'stripe',
                    'position': 'top', 'target_class': 0}
    }

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(test_image.permute(1, 2, 0))
    axes[0].set_title('Original')
    axes[0].axis('off')

    for idx, (name, config) in enumerate(triggers.items(), 1):
        trigger = create_trigger(config)
        poisoned, _ = trigger.apply(test_image, 5)
        axes[idx].imshow(poisoned.permute(1, 2, 0))
        axes[idx].set_title(name.capitalize())
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('backdoor_triggers_demo.png', dpi=150, bbox_inches='tight')
    print("Saved trigger demo to backdoor_triggers_demo.png")
