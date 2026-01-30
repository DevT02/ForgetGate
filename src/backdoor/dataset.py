"""
Backdoor dataset wrapper for poisoned training
Extends standard datasets with backdoor trigger injection
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List
import numpy as np

from .triggers import create_trigger, BackdoorTrigger


class BackdoorDataset(Dataset):
    """Wrapper that injects backdoor triggers into clean dataset"""

    def __init__(self,
                 clean_dataset: Dataset,
                 trigger_config: Dict,
                 poison_ratio: float = 0.1,
                 poison_indices: Optional[List[int]] = None,
                 seed: int = 42):
        """
        Args:
            clean_dataset: Base clean dataset
            trigger_config: Configuration for backdoor trigger
            poison_ratio: Fraction of samples to poison (if poison_indices not provided)
            poison_indices: Explicit list of indices to poison (overrides poison_ratio)
            seed: Random seed for reproducible poisoning
        """
        self.clean_dataset = clean_dataset
        self.trigger = create_trigger(trigger_config)
        self.poison_ratio = poison_ratio
        self.seed = seed

        # Determine which samples to poison
        if poison_indices is not None:
            self.poison_indices = set(poison_indices)
        else:
            # Randomly select samples to poison
            rng = np.random.RandomState(seed)
            n_samples = len(clean_dataset)
            n_poison = int(n_samples * poison_ratio)
            self.poison_indices = set(rng.choice(n_samples, n_poison, replace=False))

        print(f"[BackdoorDataset] Poisoning {len(self.poison_indices)} samples "
              f"({len(self.poison_indices)/len(clean_dataset)*100:.1f}%) "
              f"with {self.trigger.trigger_type} trigger -> class {self.trigger.target_class}")

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        image, label = self.clean_dataset[idx]

        # Apply trigger if this sample should be poisoned
        if idx in self.poison_indices:
            image, label = self.trigger.apply(image, label)

        return image, label

    def get_clean_sample(self, idx):
        """Get original clean sample without trigger"""
        return self.clean_dataset[idx]


class CleanTestDataset(Dataset):
    """Dataset that only returns clean samples (no backdoor triggers)"""

    def __init__(self, clean_dataset: Dataset):
        self.clean_dataset = clean_dataset

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        return self.clean_dataset[idx]


class TriggeredTestDataset(Dataset):
    """Dataset that applies trigger to ALL samples (for attack success rate testing)"""

    def __init__(self,
                 clean_dataset: Dataset,
                 trigger_config: Dict,
                 original_labels: bool = False):
        """
        Args:
            clean_dataset: Base clean dataset
            trigger_config: Configuration for backdoor trigger
            original_labels: If True, keep original labels; if False, use target class
        """
        self.clean_dataset = clean_dataset
        self.trigger = create_trigger(trigger_config)
        self.original_labels = original_labels

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        image, label = self.clean_dataset[idx]

        # Apply trigger
        triggered_image, target_label = self.trigger.apply(image, label)

        # Return original or target label
        final_label = label if self.original_labels else target_label

        return triggered_image, final_label


def create_backdoor_splits(clean_train: Dataset,
                           clean_test: Dataset,
                           trigger_config: Dict,
                           poison_ratio: float = 0.1,
                           seed: int = 42) -> Dict[str, Dataset]:
    """
    Create all necessary splits for backdoor experiments

    Returns:
        Dict with keys:
            'poisoned_train': Training set with backdoor samples
            'clean_test': Clean test set (for main task accuracy)
            'triggered_test': Test set with triggers on all samples (for ASR)
            'backdoor_only_test': Only backdoored samples from train (for verification)
    """
    # Create poisoned training set
    poisoned_train = BackdoorDataset(
        clean_train,
        trigger_config,
        poison_ratio=poison_ratio,
        seed=seed
    )

    # Clean test set (evaluate main task performance)
    clean_test_wrapped = CleanTestDataset(clean_test)

    # Triggered test set (evaluate attack success rate)
    triggered_test = TriggeredTestDataset(
        clean_test,
        trigger_config,
        original_labels=False  # All samples labeled with target class
    )

    return {
        'poisoned_train': poisoned_train,
        'clean_test': clean_test_wrapped,
        'triggered_test': triggered_test
    }


if __name__ == "__main__":
    # Test with CIFAR-10
    from torchvision import datasets, transforms

    # Load clean CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    clean_train = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    clean_test = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create backdoor splits
    trigger_config = {
        'type': 'patch',
        'patch_size': 4,
        'position': 'bottom_right',
        'pattern': 'checkerboard',
        'target_class': 0  # airplane
    }

    splits = create_backdoor_splits(
        clean_train,
        clean_test,
        trigger_config,
        poison_ratio=0.1,
        seed=42
    )

    print(f"\nDataset sizes:")
    print(f"  Poisoned train: {len(splits['poisoned_train'])}")
    print(f"  Clean test: {len(splits['clean_test'])}")
    print(f"  Triggered test: {len(splits['triggered_test'])}")

    # Visualize some poisoned samples
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for i in range(5):
        # Get poisoned sample
        idx = list(splits['poisoned_train'].poison_indices)[i]
        clean_img, clean_label = splits['poisoned_train'].get_clean_sample(idx)
        poison_img, poison_label = splits['poisoned_train'][idx]

        axes[0, i].imshow(clean_img.permute(1, 2, 0))
        axes[0, i].set_title(f'Clean: {clean_label}')
        axes[0, i].axis('off')

        axes[1, i].imshow(poison_img.permute(1, 2, 0))
        axes[1, i].set_title(f'Poisoned: {poison_label}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('backdoor_dataset_demo.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to backdoor_dataset_demo.png")
