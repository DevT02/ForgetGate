"""
Inference-time detection of VPT resurrection attacks
Detects when input has been adversarially prompted to resurrect forgotten knowledge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Result of VPT detection"""
    is_attack: bool
    confidence: float
    detection_score: float
    threshold: float
    method: str


class VPTDetector(nn.Module):
    """Detector for VPT resurrection attacks at inference time"""

    def __init__(self,
                 embedding_dim: int = 192,
                 detection_method: str = 'embedding_distance',
                 threshold: Optional[float] = None):
        """
        Args:
            embedding_dim: Model embedding dimension
            detection_method: Detection strategy ('embedding_distance', 'magnitude', 'entropy')
            threshold: Detection threshold (auto-calibrated if None)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.detection_method = detection_method
        self.threshold = threshold

        # Learnable reference embedding for clean inputs
        self.register_buffer('clean_embedding_mean', torch.zeros(embedding_dim))
        self.register_buffer('clean_embedding_std', torch.ones(embedding_dim))

        # Statistics
        self.register_buffer('n_clean_samples', torch.tensor(0))
        self.calibrated = False

    def calibrate(self, clean_embeddings: torch.Tensor, percentile: float = 95.0):
        """
        Calibrate detector on clean samples

        Args:
            clean_embeddings: [N, D] embeddings from clean inputs
            percentile: Detection threshold percentile (e.g., 95 = flag top 5%)
        """
        # Update clean distribution statistics
        self.clean_embedding_mean = clean_embeddings.mean(dim=0)
        self.clean_embedding_std = clean_embeddings.std(dim=0)
        self.n_clean_samples = torch.tensor(len(clean_embeddings))

        # Compute detection scores on clean samples
        with torch.no_grad():
            clean_scores = self._compute_detection_score(clean_embeddings)

        # Set threshold at percentile
        self.threshold = float(np.percentile(clean_scores.cpu().numpy(), percentile))
        self.calibrated = True

        print(f"[VPTDetector] Calibrated on {len(clean_embeddings)} clean samples")
        print(f"  Method: {self.detection_method}")
        print(f"  Threshold ({percentile}th percentile): {self.threshold:.6f}")

    def _compute_detection_score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute detection score based on chosen method

        Args:
            embeddings: [N, D] input embeddings

        Returns:
            scores: [N] detection scores (higher = more suspicious)
        """
        if self.detection_method == 'embedding_distance':
            # Mahalanobis distance from clean distribution
            normalized = (embeddings - self.clean_embedding_mean) / (self.clean_embedding_std + 1e-6)
            scores = torch.norm(normalized, p=2, dim=1)

        elif self.detection_method == 'magnitude':
            # L2 norm of embeddings (VPT prompts may have unusual magnitudes)
            scores = torch.norm(embeddings, p=2, dim=1)

        elif self.detection_method == 'entropy':
            # Entropy of softmax over embedding dimensions (unusual for VPT)
            probs = F.softmax(embeddings, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            # Invert: low entropy is suspicious
            scores = -entropy

        else:
            raise ValueError(f"Unknown detection method: {self.detection_method}")

        return scores

    def detect(self, embeddings: torch.Tensor) -> DetectionResult:
        """
        Detect VPT resurrection attack from embeddings

        Args:
            embeddings: [N, D] or [D] input embeddings

        Returns:
            DetectionResult with detection outcome
        """
        if not self.calibrated:
            raise RuntimeError("Detector not calibrated. Call calibrate() first.")

        # Handle single embedding
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        # Compute detection score
        scores = self._compute_detection_score(embeddings)
        score = scores.mean().item()  # Average over batch

        # Detect
        is_attack = score > self.threshold
        confidence = abs(score - self.threshold) / self.threshold

        return DetectionResult(
            is_attack=is_attack,
            confidence=min(confidence, 1.0),
            detection_score=score,
            threshold=self.threshold,
            method=self.detection_method
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, DetectionResult]:
        """
        Forward pass with detection

        Args:
            x: Input tensor (already embedded)

        Returns:
            x: Pass-through input
            detection: Detection result
        """
        detection = self.detect(x)
        return x, detection


class EnsembleVPTDetector(nn.Module):
    """Ensemble of multiple VPT detectors for robust detection"""

    def __init__(self, embedding_dim: int = 192):
        super().__init__()

        # Create multiple detectors with different methods
        self.detectors = nn.ModuleList([
            VPTDetector(embedding_dim, 'embedding_distance'),
            VPTDetector(embedding_dim, 'magnitude'),
            VPTDetector(embedding_dim, 'entropy'),
        ])

    def calibrate(self, clean_embeddings: torch.Tensor, percentile: float = 95.0):
        """Calibrate all detectors"""
        for detector in self.detectors:
            detector.calibrate(clean_embeddings, percentile)

    def detect(self, embeddings: torch.Tensor) -> DetectionResult:
        """Ensemble detection: flag if majority of detectors agree"""
        results = [detector.detect(embeddings) for detector in self.detectors]

        # Majority vote
        votes = sum(r.is_attack for r in results)
        is_attack = votes >= (len(results) / 2)

        # Average confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)
        avg_score = sum(r.detection_score for r in results) / len(results)

        return DetectionResult(
            is_attack=is_attack,
            confidence=avg_confidence,
            detection_score=avg_score,
            threshold=results[0].threshold,  # Use first detector's threshold
            method='ensemble'
        )


def extract_embeddings_from_model(model: nn.Module,
                                  dataloader: torch.utils.data.DataLoader,
                                  device: torch.device,
                                  max_batches: Optional[int] = None) -> torch.Tensor:
    """
    Extract embeddings from model for calibration

    Args:
        model: Model to extract from (should have .get_embedding() method)
        dataloader: Data to extract embeddings from
        device: Device to run on
        max_batches: Max batches to process (for efficiency)

    Returns:
        embeddings: [N, D] stacked embeddings
    """
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            images = images.to(device)

            # Extract embeddings (assumes model has method to get pre-classifier embeddings)
            if hasattr(model, 'get_embedding'):
                embeddings = model.get_embedding(images)
            else:
                # Fallback: use penultimate layer output
                # This is model-specific and may need adjustment
                embeddings = model(images)

            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


if __name__ == "__main__":
    # Test detector
    print("Testing VPT Detector...")

    embedding_dim = 192

    # Generate synthetic clean embeddings
    n_clean = 1000
    clean_embeddings = torch.randn(n_clean, embedding_dim) * 0.5

    # Generate synthetic VPT-attacked embeddings (shifted distribution)
    n_attack = 100
    attack_embeddings = torch.randn(n_attack, embedding_dim) * 0.5 + 2.0  # Shifted mean

    # Test single detector
    print("\n[Testing Single Detector]")
    detector = VPTDetector(embedding_dim, detection_method='embedding_distance')
    detector.calibrate(clean_embeddings, percentile=95.0)

    # Test on clean samples
    clean_result = detector.detect(clean_embeddings[:10])
    print(f"Clean samples - Attack detected: {clean_result.is_attack}, "
          f"Score: {clean_result.detection_score:.4f}")

    # Test on attack samples
    attack_result = detector.detect(attack_embeddings[:10])
    print(f"Attack samples - Attack detected: {attack_result.is_attack}, "
          f"Score: {attack_result.detection_score:.4f}")

    # Test ensemble detector
    print("\n[Testing Ensemble Detector]")
    ensemble = EnsembleVPTDetector(embedding_dim)
    ensemble.calibrate(clean_embeddings, percentile=95.0)

    clean_result_ens = ensemble.detect(clean_embeddings[:10])
    print(f"Clean samples (ensemble) - Attack detected: {clean_result_ens.is_attack}")

    attack_result_ens = ensemble.detect(attack_embeddings[:10])
    print(f"Attack samples (ensemble) - Attack detected: {attack_result_ens.is_attack}")

    # Compute detection metrics
    print("\n[Detection Performance]")
    all_clean_results = [detector.detect(emb.unsqueeze(0)).is_attack
                         for emb in clean_embeddings[:100]]
    all_attack_results = [detector.detect(emb.unsqueeze(0)).is_attack
                          for emb in attack_embeddings]

    false_positive_rate = sum(all_clean_results) / len(all_clean_results)
    true_positive_rate = sum(all_attack_results) / len(all_attack_results)

    print(f"False Positive Rate: {false_positive_rate:.2%}")
    print(f"True Positive Rate: {true_positive_rate:.2%}")
