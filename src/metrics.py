"""
Evaluation metrics for ForgetGate-V
Includes KL divergence, logit margins, and other unlearning metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
import numpy as np


def compute_kl_divergence(model1: nn.Module, model2: nn.Module,
                         data_loader: DataLoader, device: str = "cuda",
                         temperature: float = 1.0) -> Dict[str, float]:
    """
    Compute KL divergence between two models' output distributions

    This measures how similar two models are in their predictions.
    For unlearning: KL(unlearned || oracle) should be small if unlearning = true forgetting

    Args:
        model1: First model (e.g., unlearned model)
        model2: Second model (e.g., retraining oracle)
        data_loader: Data to compare on (typically retain set)
        device: Device to run on
        temperature: Temperature for softmax (higher = smoother distributions)

    Returns:
        Dictionary with KL divergence metrics
    """
    model1.eval()
    model2.eval()

    kl_sum = 0.0
    kl_per_sample = []
    reverse_kl_sum = 0.0  # KL(model2 || model1)
    js_divergence_sum = 0.0  # Jensen-Shannon divergence (symmetric)
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # Get logits from both models
            logits1 = model1(inputs)
            logits2 = model2(inputs)

            # Convert to probability distributions with temperature
            log_probs1 = F.log_softmax(logits1 / temperature, dim=1)
            probs1 = F.softmax(logits1 / temperature, dim=1)
            log_probs2 = F.log_softmax(logits2 / temperature, dim=1)
            probs2 = F.softmax(logits2 / temperature, dim=1)

            # KL(P || Q) = sum(P * log(P/Q))
            # In PyTorch: F.kl_div expects log_Q and P as inputs
            kl = F.kl_div(log_probs2, probs1, reduction='none').sum(dim=1)
            reverse_kl = F.kl_div(log_probs1, probs2, reduction='none').sum(dim=1)

            # Jensen-Shannon divergence: symmetric version of KL
            # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
            m_probs = 0.5 * (probs1 + probs2)
            log_m_probs = torch.log(m_probs + 1e-12)
            js = 0.5 * F.kl_div(log_m_probs, probs1, reduction='none').sum(dim=1) + \
                 0.5 * F.kl_div(log_m_probs, probs2, reduction='none').sum(dim=1)

            kl_sum += kl.sum().item()
            reverse_kl_sum += reverse_kl.sum().item()
            js_divergence_sum += js.sum().item()
            kl_per_sample.extend(kl.cpu().numpy().tolist())
            total_samples += batch_size

    results = {
        'kl_divergence': kl_sum / total_samples,  # Average KL(model1 || model2)
        'reverse_kl_divergence': reverse_kl_sum / total_samples,  # KL(model2 || model1)
        'js_divergence': js_divergence_sum / total_samples,  # Symmetric JS divergence
        'kl_std': np.std(kl_per_sample),  # Variability across samples
        'kl_max': np.max(kl_per_sample),  # Worst case divergence
        'total_samples': total_samples,
    }

    return results


def compute_logit_margins(model: nn.Module, data_loader: DataLoader,
                          target_class: int, device: str = "cuda") -> Dict[str, float]:
    """
    Compute logit margins for target class

    Margin = logit[target] - max(logit[other classes])
    Negative margin means model doesn't predict target class

    Args:
        model: Model to evaluate
        data_loader: Data loader (typically forget class samples)
        target_class: Class to compute margins for
        device: Device to run on

    Returns:
        Dictionary with margin statistics
    """
    model.eval()

    margins = []
    target_logits = []
    max_other_logits = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            logits = model(inputs)

            # Get logit for target class
            target_logit = logits[:, target_class]

            # Get max logit among other classes
            # Create mask to exclude target class
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[:, target_class] = False
            other_logits = logits[mask].view(logits.size(0), -1)
            max_other_logit = other_logits.max(dim=1)[0]

            # Margin = target - max_other
            margin = target_logit - max_other_logit

            margins.extend(margin.cpu().numpy().tolist())
            target_logits.extend(target_logit.cpu().numpy().tolist())
            max_other_logits.extend(max_other_logit.cpu().numpy().tolist())

    margins = np.array(margins)

    return {
        'mean_margin': float(np.mean(margins)),
        'std_margin': float(np.std(margins)),
        'min_margin': float(np.min(margins)),
        'max_margin': float(np.max(margins)),
        'negative_margin_ratio': float((margins < 0).mean()),  # Fraction correctly "forgotten"
        'mean_target_logit': float(np.mean(target_logits)),
        'mean_max_other_logit': float(np.mean(max_other_logits)),
    }


def compute_prediction_agreement(model1: nn.Module, model2: nn.Module,
                                 data_loader: DataLoader, device: str = "cuda") -> float:
    """
    Compute fraction of samples where two models agree on predictions

    Args:
        model1: First model
        model2: Second model
        data_loader: Data loader
        device: Device to run on

    Returns:
        Agreement rate (0-1)
    """
    model1.eval()
    model2.eval()

    agreements = 0
    total = 0

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)

            pred1 = model1(inputs).argmax(dim=1)
            pred2 = model2(inputs).argmax(dim=1)

            agreements += (pred1 == pred2).sum().item()
            total += inputs.size(0)

    return agreements / total if total > 0 else 0.0


def compute_representation_similarity(model1: nn.Module, model2: nn.Module,
                                     data_loader: DataLoader, device: str = "cuda",
                                     layer_name: str = "penultimate") -> Dict[str, float]:
    """
    Compute representation similarity between two models using CKA
    (Centered Kernel Alignment)

    This measures how similar the internal representations are.

    Args:
        model1: First model
        model2: Second model
        data_loader: Data loader
        device: Device to run on
        layer_name: Which layer to compare (not implemented yet - uses final features)

    Returns:
        Dictionary with similarity metrics
    """
    # Simplified version: compare final layer features
    # Full implementation would need hooks to extract intermediate representations

    model1.eval()
    model2.eval()

    features1_list = []
    features2_list = []

    # This is a simplified version - would need hooks for intermediate layers
    # For now, we use the outputs before softmax as "features"
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)

            # Get pre-softmax logits as features
            feat1 = model1(inputs)
            feat2 = model2(inputs)

            features1_list.append(feat1.cpu())
            features2_list.append(feat2.cpu())

    features1 = torch.cat(features1_list, dim=0)  # [N, num_classes]
    features2 = torch.cat(features2_list, dim=0)  # [N, num_classes]

    # Compute centered features
    features1 = features1 - features1.mean(dim=0, keepdim=True)
    features2 = features2 - features2.mean(dim=0, keepdim=True)

    # Compute CKA
    # CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    xy = torch.matmul(features1.T, features2)  # [d1, d2]
    xx = torch.matmul(features1.T, features1)  # [d1, d1]
    yy = torch.matmul(features2.T, features2)  # [d2, d2]

    cka = (xy.norm('fro') ** 2) / (xx.norm('fro') * yy.norm('fro') + 1e-12)

    # Also compute cosine similarity (simpler metric)
    cosine_sim = F.cosine_similarity(
        features1.flatten().unsqueeze(0),
        features2.flatten().unsqueeze(0)
    ).item()

    return {
        'cka': float(cka.item()),
        'cosine_similarity': float(cosine_sim),
    }


def evaluate_unlearning_quality(unlearned_model: nn.Module,
                                oracle_model: Optional[nn.Module],
                                retain_loader: DataLoader,
                                forget_loader: DataLoader,
                                forget_class: int,
                                device: str = "cuda") -> Dict[str, any]:
    """
    Comprehensive evaluation of unlearning quality

    Compares unlearned model to retraining oracle (if available)
    and computes various metrics

    Args:
        unlearned_model: Model after unlearning
        oracle_model: Model trained without forget class (can be None)
        retain_loader: Data loader for retain set
        forget_loader: Data loader for forget set
        forget_class: The forgotten class index
        device: Device to run on

    Returns:
        Dictionary with comprehensive metrics
    """
    results = {}

    # 1. Logit margins on forget class
    forget_margins = compute_logit_margins(
        unlearned_model, forget_loader, forget_class, device
    )
    results['forget_margins'] = forget_margins

    # 2. Compare to oracle if available
    if oracle_model is not None:
        # KL divergence on retain set
        kl_metrics = compute_kl_divergence(
            unlearned_model, oracle_model, retain_loader, device
        )
        results['kl_to_oracle'] = kl_metrics

        # Prediction agreement
        agreement = compute_prediction_agreement(
            unlearned_model, oracle_model, retain_loader, device
        )
        results['oracle_agreement'] = agreement

        # Representation similarity
        rep_sim = compute_representation_similarity(
            unlearned_model, oracle_model, retain_loader, device
        )
        results['oracle_similarity'] = rep_sim

    return results


if __name__ == "__main__":
    print("Metrics module for unlearning evaluation")
    print("Includes: KL divergence, logit margins, CKA, and more")
