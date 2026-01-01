"""
Data loading and preprocessing for ForgetGate-V
Supports CIFAR-10, MNIST with class filtering for unlearning
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import yaml
import os
from itertools import cycle


class ClassFilteredDataset(Dataset):
    """Dataset wrapper that filters samples by class"""

    def __init__(self, base_dataset: Dataset, include_classes: Optional[List[int]] = None,
                 exclude_classes: Optional[List[int]] = None):
        self.base_dataset = base_dataset
        self.include_classes = include_classes
        self.exclude_classes = exclude_classes

        # Build index of valid samples - use targets if available for better performance
        self.valid_indices = []
        if hasattr(base_dataset, 'targets') and base_dataset.targets is not None:
            # Fast path: use targets attribute (available in CIFAR10, MNIST, etc.)
            for idx, label in enumerate(base_dataset.targets):
                label = int(label)  # Coerce tensor labels to int (needed for MNIST)
                if self._is_valid_class(label):
                    self.valid_indices.append(idx)
        else:
            # Slow path: iterate through dataset (fallback for other datasets)
            for idx in range(len(base_dataset)):
                _, label = base_dataset[idx]
                if self._is_valid_class(label):
                    self.valid_indices.append(idx)

    def _is_valid_class(self, label: int) -> bool:
        if self.include_classes is not None:
            return label in self.include_classes
        if self.exclude_classes is not None:
            return label not in self.exclude_classes
        return True

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        return self.base_dataset[actual_idx]


class DataManager:
    """Manages dataset loading and preprocessing"""

    def __init__(self, config_path: str = "configs/data.yaml", data_dir: Optional[str] = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data_dir = data_dir or './data'

    def get_transforms(self, dataset_name: str, split: str = "train", use_pretrained: bool = True,
                      apply_imagenet_norm: bool = True) -> transforms.Compose:
        """Get appropriate transforms for dataset and split"""
        dataset_config = self.config[dataset_name]

        # Base transforms
        transform_list = []

        if split == "train":
            # Training augmentations
            if use_pretrained:
                if dataset_name == "cifar10":
                    # For CIFAR-10 with pretrained models: resize to 224
                    transform_list.extend([
                        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                        transforms.RandomHorizontalFlip(),
                    ])
                elif dataset_name == "mnist":
                    # For MNIST with pretrained models: resize to 224
                    transform_list.extend([
                        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                        transforms.RandomHorizontalFlip(),
                    ])
                    # Channel conversion happens after ToTensor
                else:
                    # Standard ImageNet-style training
                    transform_list.extend([
                        transforms.RandomResizedCrop(dataset_config['image_size']),
                        transforms.RandomHorizontalFlip(),
                    ])
            elif dataset_config.get('image_size', 224) == 32:
                # Original CIFAR-10/MNIST transforms for non-pretrained models
                transform_list.extend([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])
            else:
                # Standard ImageNet-style training
                transform_list.extend([
                    transforms.RandomResizedCrop(dataset_config['image_size']),
                    transforms.RandomHorizontalFlip(),
                ])
        else:
            # Validation/test transforms
            if use_pretrained:
                if dataset_name == "cifar10":
                    # For CIFAR-10 with pretrained models: resize to 224
                    transform_list.extend([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                    ])
                elif dataset_name == "mnist":
                    # For MNIST with pretrained models: resize to 224 and convert to 3 channels
                    transform_list.extend([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                    ])
                    # Channel conversion happens after ToTensor
                else:
                    # Standard ImageNet-style evaluation
                    transform_list.extend([
                        transforms.Resize(256),
                        transforms.CenterCrop(dataset_config['image_size']),
                    ])
            elif dataset_config.get('image_size', 224) == 32:
                # Original CIFAR-10/MNIST eval transforms
                transform_list.append(transforms.Resize(32))
            else:
                # Standard ImageNet-style evaluation
                transform_list.extend([
                    transforms.Resize(256),
                    transforms.CenterCrop(dataset_config['image_size']),
                ])

        # Always apply ToTensor
        transform_list.append(transforms.ToTensor())

        # Apply normalization based on parameters
        if apply_imagenet_norm:
            # Apply normalization (ImageNet or dataset-specific)
            if use_pretrained:
                if dataset_name == "cifar10":
                    # ImageNet normalization for CIFAR-10 with pretrained models
                    transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
                elif dataset_name == "mnist":
                    # Convert MNIST to 3 channels and use ImageNet normalization for pretrained models
                    transform_list.extend([
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1 -> 3 channels
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                else:
                    # Dataset-specific normalization
                    transform_list.append(transforms.Normalize(dataset_config['mean'], dataset_config['std']))
            else:
                # Dataset-specific normalization for non-pretrained models
                transform_list.append(transforms.Normalize(dataset_config['mean'], dataset_config['std']))
        else:
            # No normalization - for adversarial evaluation (normalization happens in model)
            if use_pretrained and dataset_name == "mnist":
                # Still need to convert MNIST to 3 channels
                transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

        return transforms.Compose(transform_list)

    def load_dataset(self, dataset_name: str, split: str = "train",
                    include_classes: Optional[List[int]] = None,
                    exclude_classes: Optional[List[int]] = None,
                    use_pretrained: bool = True, apply_imagenet_norm: bool = True) -> Dataset:
        """Load and optionally filter dataset"""

        transform = self.get_transforms(dataset_name, split, use_pretrained, apply_imagenet_norm)

        if dataset_name == "cifar10":
            if split == "train":
                dataset = torchvision.datasets.CIFAR10(
                    root=self.data_dir, train=True, download=True, transform=transform)
            else:
                dataset = torchvision.datasets.CIFAR10(
                    root=self.data_dir, train=False, download=True, transform=transform)

        elif dataset_name == "mnist":
            if split == "train":
                dataset = torchvision.datasets.MNIST(
                    root=self.data_dir, train=True, download=True, transform=transform)
            else:
                dataset = torchvision.datasets.MNIST(
                    root=self.data_dir, train=False, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Apply class filtering if specified
        if include_classes is not None or exclude_classes is not None:
            dataset = ClassFilteredDataset(dataset, include_classes, exclude_classes)

        return dataset

    def get_dataloader(self, dataset_name: str, split: str = "train", batch_size: int = 128,
                      include_classes: Optional[List[int]] = None,
                      exclude_classes: Optional[List[int]] = None,
                      num_workers: int = 4, use_pretrained: bool = True,
                      apply_imagenet_norm: bool = True) -> DataLoader:
        """Get DataLoader for dataset"""

        dataset = self.load_dataset(
            dataset_name, split,
            include_classes=include_classes,
            exclude_classes=exclude_classes,
            use_pretrained=use_pretrained,
            apply_imagenet_norm=apply_imagenet_norm
        )

        # Create balanced sampler if doing class filtering
        sampler = None
        if include_classes is not None or exclude_classes is not None:
            # For unlearning, we might want balanced sampling
            sampler = self._get_balanced_sampler(dataset, batch_size)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and split == "train"),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )

        return dataloader

    def _get_balanced_sampler(self, dataset: ClassFilteredDataset, batch_size: int):
        """Create balanced sampler for filtered datasets

        Creates a WeightedRandomSampler that balances classes in the dataset.
        This is particularly useful for unlearning where forget class might be underrepresented.
        """
        # Get labels for the filtered dataset
        if hasattr(dataset.base_dataset, 'targets'):
            # Fast path: use targets from base dataset and filter to valid indices
            all_labels = dataset.base_dataset.targets
            labels = [int(all_labels[idx]) for idx in dataset.valid_indices]
        else:
            # Slow path: iterate through filtered dataset
            labels = []
            for idx in range(len(dataset)):
                _, label = dataset[idx]
                labels.append(int(label))

        if not labels:
            return None

        # Count samples per class
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        # Calculate weights: inverse frequency for balanced sampling
        # This gives higher weight to underrepresented classes
        total_samples = len(labels)
        num_classes = len(class_counts)

        # Weight for each class: total_samples / (num_classes * class_count)
        # This ensures each class gets equal expected representation
        class_weights = {cls: total_samples / (num_classes * count)
                        for cls, count in class_counts.items()}

        # Assign weight to each sample
        sample_weights = [class_weights[label] for label in labels]

        # Create WeightedRandomSampler
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # Allow replacement for balanced sampling
        )

        return sampler

    def get_class_names(self, dataset_name: str) -> Dict[int, str]:
        """Get human-readable class names"""
        return self.config[dataset_name]['class_names']

    def get_num_classes(self, dataset_name: str) -> int:
        """Get number of classes in dataset"""
        return self.config[dataset_name]['num_classes']


def create_forget_retain_splits(dataset: Dataset, forget_class: int,
                               train_ratio: float = 0.8) -> Tuple[Subset, Subset, Subset, Subset]:
    """
    Create forget/retain splits for unlearning evaluation

    Returns:
        forget_train, retain_train, forget_test, retain_test
    """
    # Use targets attribute for faster indexing when available (CIFAR10, MNIST, etc.)
    if hasattr(dataset, 'targets') and dataset.targets is not None:
        labels = dataset.targets
        forget_indices = [i for i, label in enumerate(labels) if int(label) == forget_class]
        retain_indices = [i for i, label in enumerate(labels) if int(label) != forget_class]
    else:
        # Fallback: iterate through dataset (slower)
        forget_indices = []
        retain_indices = []

        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label == forget_class:
                forget_indices.append(idx)
            else:
                retain_indices.append(idx)

    # Shuffle indices for better splits
    import random
    random.shuffle(forget_indices)
    random.shuffle(retain_indices)

    # Split forget class
    n_forget_train = int(len(forget_indices) * train_ratio)
    forget_train_indices = forget_indices[:n_forget_train]
    forget_test_indices = forget_indices[n_forget_train:]

    # Split retain classes
    n_retain_train = int(len(retain_indices) * train_ratio)
    retain_train_indices = retain_indices[:n_retain_train]
    retain_test_indices = retain_indices[n_retain_train:]

    forget_train = Subset(dataset, forget_train_indices)
    retain_train = Subset(dataset, retain_train_indices)
    forget_test = Subset(dataset, forget_test_indices)
    retain_test = Subset(dataset, retain_test_indices)

    return forget_train, retain_train, forget_test, retain_test


class CombinedDataLoader:
    """DataLoader that yields paired (forget_batch, retain_batch) or (None, retain_batch).

    Includes forget batches for ~one pass over forget_loader, then retain-only batches.
    """

    def __init__(self, forget_loader: DataLoader, retain_loader: DataLoader, steps_per_epoch: int = None):
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.steps_per_epoch = steps_per_epoch or len(retain_loader)

    def __iter__(self):
        forget_iter = iter(self.forget_loader)
        retain_iter = iter(self.retain_loader)

        for step in range(self.steps_per_epoch):
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(self.retain_loader)
                retain_batch = next(retain_iter)

            # Only include forget batches for ~one pass over forget_loader
            if step < len(self.forget_loader):
                try:
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(self.forget_loader)
                    forget_batch = next(forget_iter)
                yield (forget_batch, retain_batch)
            else:
                yield (None, retain_batch)

    def __len__(self):
        return self.steps_per_epoch


if __name__ == "__main__":
    # Quick test
    data_manager = DataManager()

    # Test CIFAR-10 loading
    train_loader = data_manager.get_dataloader("cifar10", "train", batch_size=32)
    test_loader = data_manager.get_dataloader("cifar10", "test", batch_size=32)

    print(f"CIFAR-10 train batches: {len(train_loader)}")
    print(f"CIFAR-10 test batches: {len(test_loader)}")

    # Test class filtering (forget class 0)
    retain_loader = data_manager.get_dataloader("cifar10", "train",
                                                exclude_classes=[0], batch_size=32)  # all except class 0
    forget_loader = data_manager.get_dataloader("cifar10", "train",
                                                include_classes=[0], batch_size=32)  # only class 0

    print(f"Retain classes (exclude 0) - train batches: {len(retain_loader)}")
    print(f"Forget class 0 only - train batches: {len(forget_loader)}")
