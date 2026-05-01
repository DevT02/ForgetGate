"""
Universal input-space patch attack for ForgetGate.

Trains a fixed-location image patch against one or more target models, with
optional oracle-contrastive regularization and retain-side KL preservation.
"""

import itertools
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.normalize import create_imagenet_normalizer


def _as_model_list(models: Union[nn.Module, Sequence[nn.Module]]) -> List[nn.Module]:
    if isinstance(models, (list, tuple)):
        return list(models)
    return [models]


def _split_normalized_model(model: nn.Module) -> Tuple[nn.Module, nn.Module]:
    if isinstance(model, nn.Sequential):
        modules = list(model.children())
        if len(modules) >= 2:
            return modules[0], modules[1]
    return nn.Identity(), model


def _resolve_feature_model(model: nn.Module) -> Optional[nn.Module]:
    queue = [model]
    seen = set()
    fallback = None

    while queue:
        current = queue.pop(0)
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        if hasattr(current, "forward_features"):
            if fallback is None:
                fallback = current
            if hasattr(current, "blocks"):
                return current

        for attr in ("module", "base_model", "model"):
            child = getattr(current, attr, None)
            if isinstance(child, nn.Module):
                queue.append(child)

        if hasattr(current, "get_base_model"):
            try:
                child = current.get_base_model()
            except TypeError:
                child = None
            if isinstance(child, nn.Module):
                queue.append(child)

        for child in current.children():
            if isinstance(child, nn.Module):
                queue.append(child)

    return fallback


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, dim=1)


def _extract_features_and_block_reps(
    wrapped_model: nn.Module,
    inputs: torch.Tensor,
    block_indices: Sequence[int],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    if isinstance(wrapped_model, UniversalPatchModel):
        inputs = wrapped_model.apply_patch(inputs)
        wrapped_model = wrapped_model.base_model

    normalizer, inner_model = _split_normalized_model(wrapped_model)
    normalized_inputs = normalizer(inputs)
    feature_model = _resolve_feature_model(inner_model)
    if feature_model is None:
        raise ValueError("Universal patch feature guidance requires a model with forward_features support.")

    block_outputs: Dict[int, torch.Tensor] = {}
    handles = []
    blocks = getattr(feature_model, "blocks", None)
    if blocks is not None:
        for block_idx in sorted(set(int(idx) for idx in block_indices)):
            if block_idx < 0 or block_idx >= len(blocks):
                continue
            handles.append(
                blocks[block_idx].register_forward_hook(
                    lambda _module, _inputs, output, idx=block_idx: block_outputs.__setitem__(idx, output[0] if isinstance(output, (tuple, list)) else output)
                )
            )

    try:
        features = feature_model.forward_features(normalized_inputs)
    finally:
        for handle in handles:
            handle.remove()

    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim == 3:
        final_features = features[:, 0]
    elif features.ndim > 2:
        final_features = features.flatten(start_dim=1)
    else:
        final_features = features

    block_reps: Dict[int, torch.Tensor] = {}
    for block_idx, tokens in block_outputs.items():
        if tokens.ndim == 3 and tokens.size(1) > 1:
            cls_token = tokens[:, 0]
            patch_mean = tokens[:, 1:].mean(dim=1)
            block_reps[block_idx] = _normalize_features(cls_token + patch_mean)
        else:
            block_reps[block_idx] = _normalize_features(tokens.flatten(start_dim=1))

    return final_features, block_outputs, block_reps


def _fit_binary_probe(
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-2,
) -> Dict[str, torch.Tensor]:
    device = positive_features.device
    x = torch.cat([positive_features, negative_features], dim=0)
    x = _normalize_features(x)
    y = torch.cat(
        [
            torch.ones(positive_features.size(0), 1, device=device),
            torch.zeros(negative_features.size(0), 1, device=device),
        ],
        dim=0,
    )

    probe = nn.Linear(x.size(1), 1).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    for _ in range(int(epochs)):
        logits = probe(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = probe(x)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        train_acc = float((preds == y).float().mean().item())

    return {
        "weight": probe.weight.detach().clone(),
        "bias": probe.bias.detach().clone(),
        "train_acc": train_acc,
    }


def _apply_binary_probe(features: torch.Tensor, probe_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    features = _normalize_features(features)
    return F.linear(features, probe_state["weight"], probe_state["bias"]).squeeze(-1)


def _total_variation(patch: torch.Tensor) -> torch.Tensor:
    tv_h = (patch[:, :, 1:, :] - patch[:, :, :-1, :]).abs().mean()
    tv_w = (patch[:, :, :, 1:] - patch[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def _targeted_margin_loss(
    logits: torch.Tensor,
    target_labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    target_logits = logits.gather(1, target_labels.view(-1, 1)).squeeze(1)
    masked_logits = logits.clone()
    masked_logits.scatter_(1, target_labels.view(-1, 1), float("-inf"))
    max_other_logits = masked_logits.max(dim=1).values
    return F.relu(float(margin) + max_other_logits - target_logits).mean()


def _aggregate_model_losses(
    losses: Sequence[torch.Tensor],
    mode: str = "mean",
    temperature: float = 10.0,
) -> torch.Tensor:
    if not losses:
        raise ValueError("Expected at least one loss tensor to aggregate.")
    if len(losses) == 1:
        return losses[0]

    stacked = torch.stack(list(losses))
    mode = str(mode).lower()
    if mode == "mean":
        return stacked.mean()
    if mode in ("max", "worst", "dro"):
        return stacked.max()
    if mode in ("logsumexp", "lse", "softmax"):
        tau = max(float(temperature), 1e-6)
        return torch.logsumexp(stacked * tau, dim=0) / tau
    raise ValueError(f"Unsupported loss aggregation mode: {mode}")


def _aggregate_target_success(
    values: Sequence[torch.Tensor],
    mode: str = "mean",
    temperature: float = 10.0,
) -> torch.Tensor:
    if not values:
        raise ValueError("Expected at least one target-success tensor to aggregate.")
    if len(values) == 1:
        return values[0]

    stacked = torch.stack(list(values))
    mode = str(mode).lower()
    if mode == "mean":
        return stacked.mean()
    if mode in ("max", "worst", "dro"):
        return stacked.min()
    if mode in ("logsumexp", "lse", "softmax"):
        tau = max(float(temperature), 1e-6)
        return -torch.logsumexp(-stacked * tau, dim=0) / tau
    raise ValueError(f"Unsupported target-success aggregation mode: {mode}")


class UniversalPatchModel(nn.Module):
    """Wrap a model with a learnable fixed-location RGB patch."""

    def __init__(self, base_model: nn.Module, patch_config: Dict):
        super().__init__()
        self.base_model = base_model
        self.patch_size = int(patch_config.get("patch_size", 32))
        self.position = patch_config.get("position", "bottom_right")
        self.opacity = float(patch_config.get("opacity", 1.0))
        self.random_position = bool(patch_config.get("random_position", False))
        self.random_position_train_only = bool(patch_config.get("random_position_train_only", self.random_position))
        init_strategy = str(patch_config.get("init_strategy", "random"))
        init_scale = float(patch_config.get("init_scale", 0.01))

        device = next(base_model.parameters()).device
        patch_shape = (1, 3, self.patch_size, self.patch_size)
        self.patch_logits = nn.Parameter(torch.zeros(patch_shape, device=device))
        if init_strategy == "random":
            nn.init.normal_(self.patch_logits, mean=0.0, std=init_scale)
        elif init_strategy == "zeros":
            nn.init.zeros_(self.patch_logits)
        else:
            raise ValueError(f"Unsupported patch init strategy: {init_strategy}")

    def get_patch(self) -> torch.Tensor:
        return torch.sigmoid(self.patch_logits)

    def _resolve_position(self, height: int, width: int) -> Tuple[int, int]:
        if isinstance(self.position, str):
            if self.position == "bottom_right":
                return height - self.patch_size, width - self.patch_size
            if self.position == "top_left":
                return 0, 0
            if self.position == "center":
                return (height - self.patch_size) // 2, (width - self.patch_size) // 2
            raise ValueError(f"Unsupported patch position string: {self.position}")

        if isinstance(self.position, (list, tuple)) and len(self.position) == 2:
            return int(self.position[0]), int(self.position[1])

        raise ValueError(f"Unsupported patch position value: {self.position}")

    def apply_patch(self, inputs: torch.Tensor) -> torch.Tensor:
        patched = inputs.clone()
        batch_size, _, height, width = patched.shape
        patch = self.get_patch().to(device=patched.device, dtype=patched.dtype)

        use_random_position = self.random_position and (
            self.training or not self.random_position_train_only
        )
        if use_random_position:
            max_top = max(height - self.patch_size, 0)
            max_left = max(width - self.patch_size, 0)
            tops = (
                torch.randint(0, max_top + 1, (batch_size,), device=patched.device)
                if max_top > 0
                else torch.zeros(batch_size, dtype=torch.long, device=patched.device)
            )
            lefts = (
                torch.randint(0, max_left + 1, (batch_size,), device=patched.device)
                if max_left > 0
                else torch.zeros(batch_size, dtype=torch.long, device=patched.device)
            )
            for batch_idx in range(batch_size):
                top = int(tops[batch_idx].item())
                left = int(lefts[batch_idx].item())
                bottom = min(height, top + self.patch_size)
                right = min(width, left + self.patch_size)
                patch_view = patch[:, :, : bottom - top, : right - left]
                original = patched[batch_idx : batch_idx + 1, :, top:bottom, left:right]
                blended = (1.0 - self.opacity) * original + self.opacity * patch_view
                patched[batch_idx : batch_idx + 1, :, top:bottom, left:right] = blended
        else:
            top, left = self._resolve_position(height, width)
            bottom = min(height, top + self.patch_size)
            right = min(width, left + self.patch_size)
            patch_view = patch[:, :, : bottom - top, : right - left]
            original = patched[:, :, top:bottom, left:right]
            blended = (1.0 - self.opacity) * original + self.opacity * patch_view
            patched[:, :, top:bottom, left:right] = blended
        return torch.clamp(patched, 0.0, 1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base_model(self.apply_patch(inputs))


class UniversalPatchAttack:
    """Train and evaluate a universal image patch on one or more target models."""

    def __init__(
        self,
        target_models: Union[nn.Module, Sequence[nn.Module]],
        oracle_model: Optional[nn.Module],
        forget_class: int,
        patch_config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.forget_class = int(forget_class)
        self.patch_config = dict(patch_config)
        self.attack_history: List[Dict] = []
        self.feature_mode = str(self.patch_config.get("feature_mode", "")).lower().strip()
        self.harp_state: Optional[Dict] = None

        target_models = _as_model_list(target_models)
        self.target_clean_models = [
            nn.Sequential(create_imagenet_normalizer(), model.to(device)).to(device)
            for model in target_models
        ]
        for model in self.target_clean_models:
            for param in model.parameters():
                param.requires_grad = False

        self.oracle_clean_model = None
        if oracle_model is not None:
            self.oracle_clean_model = nn.Sequential(
                create_imagenet_normalizer(), oracle_model.to(device)
            ).to(device)
            for param in self.oracle_clean_model.parameters():
                param.requires_grad = False

        self.patched_target_models = [
            UniversalPatchModel(model, self.patch_config).to(device)
            for model in self.target_clean_models
        ]
        for extra_model in self.patched_target_models[1:]:
            extra_model.patch_logits = self.patched_target_models[0].patch_logits

        self.patched_target_model = self.patched_target_models[0]

        self.oracle_patched_model = None
        if self.oracle_clean_model is not None:
            self.oracle_patched_model = UniversalPatchModel(
                self.oracle_clean_model, self.patch_config
            ).to(device)
            self.oracle_patched_model.patch_logits = self.patched_target_model.patch_logits

        for model in self.target_clean_models:
            model.eval()
        for model in self.patched_target_models:
            model.eval()
            model.base_model.eval()
        if self.oracle_clean_model is not None:
            self.oracle_clean_model.eval()
        if self.oracle_patched_model is not None:
            self.oracle_patched_model.eval()
            self.oracle_patched_model.base_model.eval()

    def _collect_model_features(
        self,
        model: nn.Module,
        loader: DataLoader,
        block_indices: Sequence[int],
        max_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        final_features = []
        labels = []
        block_features: Dict[int, List[torch.Tensor]] = {int(idx): [] for idx in block_indices}
        collected = 0

        model.eval()
        with torch.no_grad():
            for inputs, batch_labels in loader:
                inputs = inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                feats, _block_tokens, block_reps = _extract_features_and_block_reps(
                    model, inputs, block_indices
                )
                final_features.append(feats.detach())
                labels.append(batch_labels.detach())
                for block_idx in block_indices:
                    rep = block_reps.get(int(block_idx))
                    if rep is not None:
                        block_features[int(block_idx)].append(rep.detach())
                collected += inputs.size(0)
                if collected >= max_samples:
                    break

        if not final_features:
            raise ValueError("Failed to collect features for HARP patch guidance.")

        final_tensor = torch.cat(final_features, dim=0)[:max_samples]
        label_tensor = torch.cat(labels, dim=0)[:max_samples]
        block_tensors: Dict[int, torch.Tensor] = {}
        for block_idx, reps in block_features.items():
            if reps:
                block_tensors[block_idx] = torch.cat(reps, dim=0)[:max_samples]
        return final_tensor, label_tensor, block_tensors

    def _prepare_harp_guidance(self, forget_loader: DataLoader, retain_loader: DataLoader) -> Dict:
        block_indices = [int(idx) for idx in self.patch_config.get("token_blocks", [0, 5, 11])]
        max_forget = int(self.patch_config.get("max_forget_samples", 1024))
        max_retain = int(self.patch_config.get("max_retain_samples", 2048))
        probe_epochs = int(self.patch_config.get("probe_epochs", 200))
        probe_lr = float(self.patch_config.get("probe_lr", 1e-2))

        model_states = []
        for model in self.target_clean_models:
            forget_feats, _forget_labels, forget_blocks = self._collect_model_features(
                model, forget_loader, block_indices, max_forget
            )
            retain_feats, retain_labels, retain_blocks = self._collect_model_features(
                model, retain_loader, block_indices, max_retain
            )

            forget_feats_norm = _normalize_features(forget_feats)
            retain_feats_norm = _normalize_features(retain_feats)
            forget_proto = _normalize_features(forget_feats_norm.mean(dim=0, keepdim=True)).squeeze(0)
            retain_proto_labels = []
            retain_proto_list = []
            for label in sorted(int(lbl) for lbl in retain_labels.unique().tolist() if int(lbl) != self.forget_class):
                mask = retain_labels == label
                if mask.any():
                    retain_proto_labels.append(label)
                    retain_proto_list.append(
                        _normalize_features(retain_feats_norm[mask].mean(dim=0, keepdim=True)).squeeze(0)
                    )

            final_probe = _fit_binary_probe(
                positive_features=forget_feats,
                negative_features=retain_feats,
                epochs=probe_epochs,
                lr=probe_lr,
            )

            block_probes = {}
            for block_idx in block_indices:
                if block_idx not in forget_blocks or block_idx not in retain_blocks:
                    continue
                block_probes[block_idx] = _fit_binary_probe(
                    positive_features=forget_blocks[block_idx],
                    negative_features=retain_blocks[block_idx],
                    epochs=probe_epochs,
                    lr=probe_lr,
                )

            model_states.append(
                {
                    "forget_proto": forget_proto.detach(),
                    "retain_proto_labels": torch.tensor(retain_proto_labels, device=self.device, dtype=torch.long),
                    "retain_proto_matrix": torch.stack(retain_proto_list, dim=0).detach() if retain_proto_list else torch.empty(0, forget_feats.size(1), device=self.device),
                    "final_probe": final_probe,
                    "block_probes": block_probes,
                }
            )

        summary = {
            "mode": "harp",
            "block_indices": block_indices,
            "max_forget_samples": max_forget,
            "max_retain_samples": max_retain,
            "probe_epochs": probe_epochs,
            "probe_lr": probe_lr,
            "target_probe_train_acc": [float(state["final_probe"]["train_acc"]) for state in model_states],
        }
        self.harp_state = {
            "summary": summary,
            "block_indices": block_indices,
            "models": model_states,
        }
        return self.harp_state

    def _compute_harp_losses(self, inputs_forget: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.harp_state:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "loss_proto": zero,
                "loss_final_probe": zero,
                "loss_token_probe": zero,
                "loss_sink": zero,
            }

        proto_margin = float(self.patch_config.get("prototype_margin", 0.1))
        final_probe_margin = float(self.patch_config.get("final_probe_margin", 1.0))
        token_probe_margin = float(self.patch_config.get("token_probe_margin", 1.0))
        block_indices = self.harp_state["block_indices"]

        loss_proto_terms = []
        loss_final_probe_terms = []
        loss_token_probe_terms = []
        loss_sink_terms = []

        for patched_model, state in zip(self.patched_target_models, self.harp_state["models"]):
            final_features, block_tokens, block_reps = _extract_features_and_block_reps(
                patched_model, inputs_forget, block_indices
            )
            final_features_norm = _normalize_features(final_features)

            final_scores = _apply_binary_probe(final_features_norm, state["final_probe"])
            loss_final_probe_terms.append(F.relu(final_probe_margin - final_scores).mean())

            retain_proto_matrix = state["retain_proto_matrix"]
            if retain_proto_matrix.numel() > 0:
                d_forget = torch.cdist(final_features_norm, state["forget_proto"].unsqueeze(0), p=2).squeeze(1)
                d_retain = torch.cdist(final_features_norm, retain_proto_matrix, p=2).min(dim=1).values
                loss_proto_terms.append(F.relu(proto_margin + d_forget - d_retain).mean())

            for block_idx, probe_state in state["block_probes"].items():
                rep = block_reps.get(int(block_idx))
                tokens = block_tokens.get(int(block_idx))
                if rep is not None:
                    scores = _apply_binary_probe(rep, probe_state)
                    loss_token_probe_terms.append(F.relu(token_probe_margin - scores).mean())
                if tokens is not None and tokens.ndim == 3 and tokens.size(1) > 1:
                    cls_token = tokens[:, 0]
                    patch_mean = tokens[:, 1:].mean(dim=1)
                    loss_sink_terms.append((1.0 - F.cosine_similarity(cls_token, patch_mean, dim=1)).mean())

        zero = torch.tensor(0.0, device=self.device)
        return {
            "loss_proto": torch.stack(loss_proto_terms).mean() if loss_proto_terms else zero,
            "loss_final_probe": torch.stack(loss_final_probe_terms).mean() if loss_final_probe_terms else zero,
            "loss_token_probe": torch.stack(loss_token_probe_terms).mean() if loss_token_probe_terms else zero,
            "loss_sink": torch.stack(loss_sink_terms).mean() if loss_sink_terms else zero,
        }

    def _evaluate_resurrection(self, forget_loader: DataLoader) -> Dict[str, float]:
        self.patched_target_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in forget_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.patched_target_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return {
            "resurrection_acc": correct / total if total > 0 else 0.0,
            "resurrection_loss": total_loss / max(len(forget_loader), 1),
        }

    def _train_one_epoch(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        optimizer: optim.Optimizer,
        lambda_retain: float,
        temperature: float,
        oracle_contrast_weight: float,
    ) -> Dict[str, float]:
        for model in self.patched_target_models:
            model.train()
            model.base_model.eval()
        if self.oracle_patched_model is not None:
            self.oracle_patched_model.train()
            self.oracle_patched_model.base_model.eval()
        retain_iter = itertools.cycle(retain_loader)

        epoch_loss = 0.0
        loss_forget_last = 0.0
        loss_retain_last = 0.0
        loss_oracle_last = 0.0
        loss_proto_last = 0.0
        loss_final_probe_last = 0.0
        loss_token_probe_last = 0.0
        loss_sink_last = 0.0
        loss_tv_last = 0.0
        target_forget_prob_last = 0.0
        oracle_forget_prob_last = 0.0
        num_batches = 0

        prototype_weight = float(self.patch_config.get("prototype_weight", 0.0))
        final_probe_weight = float(self.patch_config.get("final_probe_weight", 0.0))
        token_probe_weight = float(self.patch_config.get("token_probe_weight", 0.0))
        sink_weight = float(self.patch_config.get("sink_weight", 0.0))
        tv_weight = float(self.patch_config.get("tv_weight", 0.0))
        forget_objective = str(self.patch_config.get("forget_objective", "cross_entropy")).lower()
        target_margin = float(self.patch_config.get("target_margin", 1.0))
        forget_aggregation = str(self.patch_config.get("forget_aggregation", "mean")).lower()
        forget_aggregation_temperature = float(self.patch_config.get("forget_aggregation_temperature", 10.0))
        oracle_contrast_margin = float(self.patch_config.get("oracle_contrast_margin", 0.05))

        for inputs_forget, labels_forget in forget_loader:
            inputs_forget = inputs_forget.to(self.device)
            labels_forget = labels_forget.to(self.device)
            inputs_retain, _ = next(retain_iter)
            inputs_retain = inputs_retain.to(self.device)

            forget_outputs = [model(inputs_forget) for model in self.patched_target_models]
            if forget_objective == "targeted_margin":
                forget_losses = [
                    _targeted_margin_loss(outputs, labels_forget, target_margin) for outputs in forget_outputs
                ]
            else:
                forget_losses = [
                    F.cross_entropy(outputs, labels_forget) for outputs in forget_outputs
                ]
            loss_forget = _aggregate_model_losses(
                forget_losses,
                mode=forget_aggregation,
                temperature=forget_aggregation_temperature,
            )

            clean_retain_outputs = []
            with torch.no_grad():
                for model in self.target_clean_models:
                    clean_retain_outputs.append(model(inputs_retain))

            patched_retain_outputs = [model(inputs_retain) for model in self.patched_target_models]
            loss_retain_terms = []
            for clean_out, patched_out in zip(clean_retain_outputs, patched_retain_outputs):
                loss_retain_terms.append(
                    F.kl_div(
                        F.log_softmax(patched_out / temperature, dim=1),
                        F.softmax(clean_out / temperature, dim=1),
                        reduction="batchmean",
                    )
                    * (temperature * temperature)
                )
            loss_retain = torch.stack(loss_retain_terms).mean()

            target_probs = [
                F.softmax(outputs, dim=1)[:, self.forget_class].mean()
                for outputs in forget_outputs
            ]
            target_forget_prob = _aggregate_target_success(
                target_probs,
                mode=forget_aggregation,
                temperature=forget_aggregation_temperature,
            )

            loss_oracle = torch.tensor(0.0, device=self.device)
            oracle_forget_prob = torch.tensor(0.0, device=self.device)
            if self.oracle_patched_model is not None and oracle_contrast_weight > 0.0:
                oracle_outputs = self.oracle_patched_model(inputs_forget)
                oracle_probs = F.softmax(oracle_outputs, dim=1)[:, self.forget_class]
                oracle_forget_prob = oracle_probs.mean()
                loss_oracle = F.relu(
                    oracle_contrast_margin + oracle_forget_prob - target_forget_prob
                )

            harp_losses = {
                "loss_proto": torch.tensor(0.0, device=self.device),
                "loss_final_probe": torch.tensor(0.0, device=self.device),
                "loss_token_probe": torch.tensor(0.0, device=self.device),
                "loss_sink": torch.tensor(0.0, device=self.device),
            }
            if self.feature_mode == "harp":
                harp_losses = self._compute_harp_losses(inputs_forget)

            tv_loss = _total_variation(self.patched_target_model.get_patch()) if tv_weight > 0.0 else torch.tensor(0.0, device=self.device)

            total_loss = (
                loss_forget
                + lambda_retain * loss_retain
                + oracle_contrast_weight * loss_oracle
                + prototype_weight * harp_losses["loss_proto"]
                + final_probe_weight * harp_losses["loss_final_probe"]
                + token_probe_weight * harp_losses["loss_token_probe"]
                + sink_weight * harp_losses["loss_sink"]
                + tv_weight * tv_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            loss_forget_last = loss_forget.item()
            loss_retain_last = loss_retain.item()
            loss_oracle_last = loss_oracle.item()
            loss_proto_last = harp_losses["loss_proto"].item()
            loss_final_probe_last = harp_losses["loss_final_probe"].item()
            loss_token_probe_last = harp_losses["loss_token_probe"].item()
            loss_sink_last = harp_losses["loss_sink"].item()
            loss_tv_last = tv_loss.item()
            target_forget_prob_last = target_forget_prob.item()
            oracle_forget_prob_last = oracle_forget_prob.item()
            num_batches += 1

        metrics = {
            "train_loss": epoch_loss / max(num_batches, 1),
            "loss_forget": loss_forget_last,
            "loss_retain": loss_retain_last,
            "loss_oracle": loss_oracle_last,
            "loss_proto": loss_proto_last,
            "loss_final_probe": loss_final_probe_last,
            "loss_token_probe": loss_token_probe_last,
            "loss_sink": loss_sink_last,
            "loss_tv": loss_tv_last,
            "target_forget_prob": target_forget_prob_last,
            "oracle_forget_prob": oracle_forget_prob_last,
            "oracle_contrast_gap": target_forget_prob_last - oracle_forget_prob_last,
        }
        return metrics

    def train_patch(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        val_forget_loader: DataLoader,
        guidance_forget_loader: Optional[DataLoader] = None,
        guidance_retain_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        lr: float = 5e-2,
        lambda_retain: float = 1.0,
        temperature: float = 1.0,
        eval_every: int = 10,
        oracle_contrast_weight: float = 1.0,
    ) -> List[Dict]:
        if self.feature_mode == "harp":
            guidance_state = self._prepare_harp_guidance(
                forget_loader=guidance_forget_loader or forget_loader,
                retain_loader=guidance_retain_loader or retain_loader,
            )
            print(f"HARP guidance enabled: {guidance_state['summary']}")

        optimizer = optim.Adam([self.patched_target_model.patch_logits], lr=lr)
        best_acc = -1.0
        best_loss = float("inf")
        best_epoch = -1
        best_state = None

        print(f"Training universal patch attack on forget={len(forget_loader.dataset)} samples")
        print(f"Retain regularization on retain={len(retain_loader.dataset)} samples")
        print(f"Validating on forget={len(val_forget_loader.dataset)} samples")

        for epoch in range(epochs):
            metrics = self._train_one_epoch(
                forget_loader=forget_loader,
                retain_loader=retain_loader,
                optimizer=optimizer,
                lambda_retain=lambda_retain,
                temperature=temperature,
                oracle_contrast_weight=oracle_contrast_weight,
            )

            if epoch % eval_every == 0 or epoch == epochs - 1:
                eval_metrics = self._evaluate_resurrection(val_forget_loader)
                metrics.update(eval_metrics)
                metrics["epoch"] = float(epoch)
                self.attack_history.append(metrics)
                print(
                    f"Epoch {epoch}: "
                    f"train_loss: {metrics['train_loss']:.4f} | "
                    f"loss_forget: {metrics['loss_forget']:.4f} | "
                    f"loss_retain: {metrics['loss_retain']:.4f} | "
                    f"loss_oracle: {metrics['loss_oracle']:.4f} | "
                    f"loss_proto: {metrics['loss_proto']:.4f} | "
                    f"loss_final_probe: {metrics['loss_final_probe']:.4f} | "
                    f"loss_token_probe: {metrics['loss_token_probe']:.4f} | "
                    f"loss_sink: {metrics['loss_sink']:.4f} | "
                    f"loss_tv: {metrics['loss_tv']:.4f} | "
                    f"target_forget_prob: {metrics['target_forget_prob']:.4f} | "
                    f"oracle_forget_prob: {metrics['oracle_forget_prob']:.4f} | "
                    f"oracle_contrast_gap: {metrics['oracle_contrast_gap']:.4f} | "
                    f"resurrection_acc: {metrics['resurrection_acc']:.4f} | "
                    f"resurrection_loss: {metrics['resurrection_loss']:.4f}"
                )

                current_acc = float(metrics["resurrection_acc"])
                current_loss = float(metrics["resurrection_loss"])
                if (
                    current_acc > best_acc
                    or (
                        abs(current_acc - best_acc) <= 1e-12
                        and current_loss < best_loss
                    )
                ):
                    best_acc = current_acc
                    best_loss = current_loss
                    best_epoch = epoch
                    best_state = {
                        "patch_logits": self.patched_target_model.patch_logits.detach().cpu().clone()
                    }
                    print(
                        f"New best checkpoint: accuracy={best_acc:.4f} "
                        f"loss={best_loss:.4f} epoch={best_epoch}"
                    )

        if best_state is None:
            best_state = {
                "patch_logits": self.patched_target_model.patch_logits.detach().cpu().clone()
            }
            best_acc = max(best_acc, 0.0)
            best_loss = float(self.attack_history[-1]["resurrection_loss"]) if self.attack_history else float("inf")
            best_epoch = int(self.attack_history[-1]["epoch"]) if self.attack_history else 0

        print(
            f"Restoring best patch with accuracy {best_acc:.4f} "
            f"loss {best_loss:.4f} from epoch {best_epoch}"
        )
        self.patched_target_model.patch_logits.data.copy_(
            best_state["patch_logits"].to(self.device)
        )
        restored_metrics = self._evaluate_resurrection(val_forget_loader)
        restored_metrics["epoch"] = float(best_epoch if best_epoch >= 0 else 0.0)
        restored_metrics["restored_best"] = 1.0
        restored_metrics["selected_by"] = "resurrection_acc_then_loss"
        self.attack_history.append(restored_metrics)
        return self.attack_history

    def save_attack_patch(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            {"patch_logits": self.patched_target_model.patch_logits.detach().cpu()},
            os.path.join(save_path, "universal_patch.pt"),
        )
        attack_info = {
            "artifact_type": "universal_patch",
            "forget_class": self.forget_class,
            "patch_config": self.patch_config,
            "attack_history": self.attack_history,
        }
        if self.harp_state is not None:
            attack_info["harp_summary"] = self.harp_state.get("summary", {})
        with open(os.path.join(save_path, "attack_info.json"), "w", encoding="utf-8") as handle:
            json.dump(attack_info, handle, indent=2)

    def load_attack_patch(self, load_path: str):
        patch_path = os.path.join(load_path, "universal_patch.pt")
        state = torch.load(patch_path, map_location=self.device)
        self.patched_target_model.patch_logits.data.copy_(state["patch_logits"].to(self.device))
        if self.oracle_patched_model is not None:
            self.oracle_patched_model.patch_logits = self.patched_target_model.patch_logits

        info_path = os.path.join(load_path, "attack_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as handle:
                info = json.load(handle)
            self.attack_history = info.get("attack_history", [])

    def get_patch_statistics(self) -> Dict[str, float]:
        patch = self.patched_target_model.get_patch().detach()
        return {
            "patch_size": self.patched_target_model.patch_size,
            "patch_mean": float(patch.mean().item()),
            "patch_std": float(patch.std().item()),
            "patch_min": float(patch.min().item()),
            "patch_max": float(patch.max().item()),
        }
