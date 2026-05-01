"""
Patch-lattice warp attack for ViT-style models.

The attack learns a smooth low-resolution flow field defined on the patch grid,
then upsamples it to the image plane and warps inputs with grid_sample.
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.normalize import create_imagenet_normalizer
from .fourier_perturbation import UniversalFourierAttack, _extract_features_and_block_reps
from .universal_patch import (
    _aggregate_model_losses,
    _aggregate_target_success,
    _as_model_list,
    _normalize_features,
    _targeted_margin_loss,
)


class PatchLatticeWarpModel(nn.Module):
    """Warp images with a smooth low-resolution flow defined on the patch grid."""

    def __init__(self, base_model: nn.Module, attack_config: Dict):
        super().__init__()
        self.base_model = base_model
        self.attack_config = dict(attack_config)
        self.image_size = int(self.attack_config.get("image_size", 224))
        self.patch_stride = int(self.attack_config.get("patch_stride", 16))
        self.flow_grid_size = int(
            self.attack_config.get("flow_grid_size", self.image_size // self.patch_stride)
        )
        self.max_flow_pixels = float(self.attack_config.get("max_flow_pixels", 3.0))
        self.flow_gain = float(self.attack_config.get("flow_gain", 2.0))

        init_scale = float(self.attack_config.get("init_scale", 1e-3))
        device = next(base_model.parameters()).device
        self.flow_logits = nn.Parameter(
            torch.zeros(1, 2, self.flow_grid_size, self.flow_grid_size, device=device)
        )
        nn.init.normal_(self.flow_logits, mean=0.0, std=init_scale)

    def get_flow_pixels(self, height: Optional[int] = None, width: Optional[int] = None) -> torch.Tensor:
        height = int(height or self.image_size)
        width = int(width or self.image_size)
        flow = self.max_flow_pixels * torch.tanh(self.flow_gain * self.flow_logits)
        flow = flow - flow.mean(dim=(2, 3), keepdim=True)
        return F.interpolate(flow, size=(height, width), mode="bicubic", align_corners=True)

    def _base_grid(self, batch_size: int, height: int, width: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def apply_perturbation(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = inputs.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Warp attack expected {self.image_size}x{self.image_size} inputs, got {height}x{width}"
            )

        flow = self.get_flow_pixels(height, width).to(device=inputs.device, dtype=inputs.dtype)
        flow = flow.expand(batch_size, -1, -1, -1)
        flow_x = 2.0 * flow[:, 0] / max(width - 1, 1)
        flow_y = 2.0 * flow[:, 1] / max(height - 1, 1)
        base_grid = self._base_grid(batch_size, height, width, inputs.device, inputs.dtype)
        sample_grid = base_grid.clone()
        sample_grid[..., 0] = sample_grid[..., 0] + flow_x
        sample_grid[..., 1] = sample_grid[..., 1] + flow_y
        warped = F.grid_sample(
            inputs,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return torch.clamp(warped, 0.0, 1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base_model(self.apply_perturbation(inputs))


class PatchLatticeWarpAttack(UniversalFourierAttack):
    """Attack patch ownership / positional geometry with a smooth lattice warp."""

    def __init__(
        self,
        target_models: Union[nn.Module, Sequence[nn.Module]],
        oracle_model: Optional[nn.Module],
        forget_class: int,
        attack_config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.forget_class = int(forget_class)
        self.attack_config = dict(attack_config)
        self.attack_history: List[Dict] = []
        self.feature_mode = str(self.attack_config.get("feature_mode", "")).lower().strip()
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

        self.perturbed_target_models = [
            PatchLatticeWarpModel(model, self.attack_config).to(device)
            for model in self.target_clean_models
        ]
        for extra_model in self.perturbed_target_models[1:]:
            extra_model.flow_logits = self.perturbed_target_models[0].flow_logits

        self.perturbed_target_model = self.perturbed_target_models[0]

        self.oracle_perturbed_model = None
        if self.oracle_clean_model is not None:
            self.oracle_perturbed_model = PatchLatticeWarpModel(
                self.oracle_clean_model, self.attack_config
            ).to(device)
            self.oracle_perturbed_model.flow_logits = self.perturbed_target_model.flow_logits

        for model in self.target_clean_models:
            model.eval()
        for model in self.perturbed_target_models:
            model.eval()
            model.base_model.eval()
        if self.oracle_clean_model is not None:
            self.oracle_clean_model.eval()
        if self.oracle_perturbed_model is not None:
            self.oracle_perturbed_model.eval()
            self.oracle_perturbed_model.base_model.eval()

    def _compute_oracle_lift_losses(self, inputs_forget: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = torch.tensor(0.0, device=self.device)
        if self.oracle_clean_model is None:
            return {"loss_final_oracle": zero, "loss_block_oracle": zero}

        block_indices = [int(idx) for idx in self.attack_config.get("token_blocks", [0, 5, 11])]
        with torch.no_grad():
            oracle_final, _oracle_tokens, oracle_block_reps = _extract_features_and_block_reps(
                self.oracle_clean_model, inputs_forget, block_indices
            )
            oracle_final = _normalize_features(oracle_final)
            oracle_block_reps = {
                int(block_idx): _normalize_features(rep)
                for block_idx, rep in oracle_block_reps.items()
            }

        final_terms = []
        block_terms = []
        for perturbed_model in self.perturbed_target_models:
            target_final, _target_tokens, target_block_reps = _extract_features_and_block_reps(
                perturbed_model, inputs_forget, block_indices
            )
            target_final = _normalize_features(target_final)
            final_terms.append((1.0 - F.cosine_similarity(target_final, oracle_final, dim=1)).mean())
            for block_idx, oracle_rep in oracle_block_reps.items():
                target_rep = target_block_reps.get(int(block_idx))
                if target_rep is None:
                    continue
                target_rep = _normalize_features(target_rep)
                block_terms.append((1.0 - F.cosine_similarity(target_rep, oracle_rep, dim=1)).mean())

        return {
            "loss_final_oracle": torch.stack(final_terms).mean() if final_terms else zero,
            "loss_block_oracle": torch.stack(block_terms).mean() if block_terms else zero,
        }

    def _evaluate_resurrection(self, forget_loader: DataLoader) -> Dict[str, float]:
        self.perturbed_target_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in forget_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.perturbed_target_model(inputs)
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
        for model in self.perturbed_target_models:
            model.train()
            model.base_model.eval()
        if self.oracle_perturbed_model is not None:
            self.oracle_perturbed_model.train()
            self.oracle_perturbed_model.base_model.eval()

        retain_iter = iter(retain_loader)
        epoch_loss = 0.0
        loss_forget_last = 0.0
        loss_retain_last = 0.0
        loss_oracle_last = 0.0
        loss_proto_last = 0.0
        loss_final_probe_last = 0.0
        loss_token_probe_last = 0.0
        loss_sink_last = 0.0
        loss_final_oracle_last = 0.0
        loss_block_oracle_last = 0.0
        loss_flow_l2_last = 0.0
        loss_flow_smooth_last = 0.0
        target_forget_prob_last = 0.0
        oracle_forget_prob_last = 0.0
        num_batches = 0

        prototype_weight = float(self.attack_config.get("prototype_weight", 0.0))
        final_probe_weight = float(self.attack_config.get("final_probe_weight", 0.0))
        token_probe_weight = float(self.attack_config.get("token_probe_weight", 0.0))
        sink_weight = float(self.attack_config.get("sink_weight", 0.0))
        final_oracle_weight = float(self.attack_config.get("final_oracle_weight", 0.0))
        block_oracle_weight = float(self.attack_config.get("block_oracle_weight", 0.0))
        flow_l2_weight = float(self.attack_config.get("flow_l2_weight", 0.0))
        flow_smooth_weight = float(self.attack_config.get("flow_smooth_weight", 0.0))
        forget_objective = str(self.attack_config.get("forget_objective", "targeted_margin")).lower()
        target_margin = float(self.attack_config.get("target_margin", 1.0))
        forget_aggregation = str(self.attack_config.get("forget_aggregation", "mean")).lower()
        forget_aggregation_temperature = float(self.attack_config.get("forget_aggregation_temperature", 10.0))
        oracle_contrast_margin = float(self.attack_config.get("oracle_contrast_margin", 0.05))

        for inputs_forget, labels_forget in forget_loader:
            inputs_forget = inputs_forget.to(self.device)
            labels_forget = labels_forget.to(self.device)
            try:
                inputs_retain, _ = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                inputs_retain, _ = next(retain_iter)
            inputs_retain = inputs_retain.to(self.device)

            forget_outputs = [model(inputs_forget) for model in self.perturbed_target_models]
            if forget_objective == "targeted_margin":
                forget_losses = [
                    _targeted_margin_loss(outputs, labels_forget, target_margin)
                    for outputs in forget_outputs
                ]
            else:
                forget_losses = [F.cross_entropy(outputs, labels_forget) for outputs in forget_outputs]
            loss_forget = _aggregate_model_losses(
                forget_losses,
                mode=forget_aggregation,
                temperature=forget_aggregation_temperature,
            )

            clean_retain_outputs = []
            with torch.no_grad():
                for model in self.target_clean_models:
                    clean_retain_outputs.append(model(inputs_retain))

            perturbed_retain_outputs = [model(inputs_retain) for model in self.perturbed_target_models]
            loss_retain_terms = []
            for clean_out, perturbed_out in zip(clean_retain_outputs, perturbed_retain_outputs):
                loss_retain_terms.append(
                    F.kl_div(
                        F.log_softmax(perturbed_out / temperature, dim=1),
                        F.softmax(clean_out / temperature, dim=1),
                        reduction="batchmean",
                    )
                    * (temperature * temperature)
                )
            loss_retain = torch.stack(loss_retain_terms).mean()

            target_probs = [F.softmax(outputs, dim=1)[:, self.forget_class].mean() for outputs in forget_outputs]
            target_forget_prob = _aggregate_target_success(
                target_probs,
                mode=forget_aggregation,
                temperature=forget_aggregation_temperature,
            )

            loss_oracle = torch.tensor(0.0, device=self.device)
            oracle_forget_prob = torch.tensor(0.0, device=self.device)
            if self.oracle_perturbed_model is not None and oracle_contrast_weight > 0.0:
                oracle_outputs = self.oracle_perturbed_model(inputs_forget)
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

            oracle_lift_losses = self._compute_oracle_lift_losses(inputs_forget)

            flow = self.perturbed_target_model.get_flow_pixels().to(self.device)
            flow_l2 = flow.pow(2).mean()
            flow_smooth = (
                (flow[:, :, 1:, :] - flow[:, :, :-1, :]).pow(2).mean()
                + (flow[:, :, :, 1:] - flow[:, :, :, :-1]).pow(2).mean()
            )

            total_loss = (
                loss_forget
                + lambda_retain * loss_retain
                + oracle_contrast_weight * loss_oracle
                + prototype_weight * harp_losses["loss_proto"]
                + final_probe_weight * harp_losses["loss_final_probe"]
                + token_probe_weight * harp_losses["loss_token_probe"]
                + sink_weight * harp_losses["loss_sink"]
                + final_oracle_weight * oracle_lift_losses["loss_final_oracle"]
                + block_oracle_weight * oracle_lift_losses["loss_block_oracle"]
                + flow_l2_weight * flow_l2
                + flow_smooth_weight * flow_smooth
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
            loss_final_oracle_last = oracle_lift_losses["loss_final_oracle"].item()
            loss_block_oracle_last = oracle_lift_losses["loss_block_oracle"].item()
            loss_flow_l2_last = flow_l2.item()
            loss_flow_smooth_last = flow_smooth.item()
            target_forget_prob_last = target_forget_prob.item()
            oracle_forget_prob_last = oracle_forget_prob.item()
            num_batches += 1

        return {
            "train_loss": epoch_loss / max(num_batches, 1),
            "loss_forget": loss_forget_last,
            "loss_retain": loss_retain_last,
            "loss_oracle": loss_oracle_last,
            "loss_proto": loss_proto_last,
            "loss_final_probe": loss_final_probe_last,
            "loss_token_probe": loss_token_probe_last,
            "loss_sink": loss_sink_last,
            "loss_final_oracle": loss_final_oracle_last,
            "loss_block_oracle": loss_block_oracle_last,
            "loss_flow_l2": loss_flow_l2_last,
            "loss_flow_smooth": loss_flow_smooth_last,
            "target_forget_prob": target_forget_prob_last,
            "oracle_forget_prob": oracle_forget_prob_last,
            "oracle_contrast_gap": target_forget_prob_last - oracle_forget_prob_last,
        }

    def train_attack(
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
            print(f"Warp guidance enabled: {guidance_state['summary']}")

        optimizer = optim.Adam([self.perturbed_target_model.flow_logits], lr=lr)
        best_acc = -1.0
        best_loss = float("inf")
        best_epoch = -1
        best_state = None

        print(f"Training warp attack on forget={len(forget_loader.dataset)} samples")
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
                    f"loss_final_oracle: {metrics['loss_final_oracle']:.4f} | "
                    f"loss_block_oracle: {metrics['loss_block_oracle']:.4f} | "
                    f"loss_flow_l2: {metrics['loss_flow_l2']:.4f} | "
                    f"loss_flow_smooth: {metrics['loss_flow_smooth']:.4f} | "
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
                        "flow_logits": self.perturbed_target_model.flow_logits.detach().cpu().clone()
                    }
                    print(
                        f"New best checkpoint: accuracy={best_acc:.4f} "
                        f"loss={best_loss:.4f} epoch={best_epoch}"
                    )

        if best_state is None:
            best_state = {
                "flow_logits": self.perturbed_target_model.flow_logits.detach().cpu().clone()
            }
            best_acc = max(best_acc, 0.0)
            best_loss = float(self.attack_history[-1]["resurrection_loss"]) if self.attack_history else float("inf")
            best_epoch = int(self.attack_history[-1]["epoch"]) if self.attack_history else 0

        print(
            f"Restoring best warp state with accuracy {best_acc:.4f} "
            f"loss {best_loss:.4f} from epoch {best_epoch}"
        )
        self.perturbed_target_model.flow_logits.data.copy_(best_state["flow_logits"].to(self.device))
        restored_metrics = self._evaluate_resurrection(val_forget_loader)
        restored_metrics["epoch"] = float(best_epoch if best_epoch >= 0 else 0.0)
        restored_metrics["restored_best"] = 1.0
        restored_metrics["selected_by"] = "resurrection_acc_then_loss"
        self.attack_history.append(restored_metrics)
        return self.attack_history

    def save_attack_artifact(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            {"flow_logits": self.perturbed_target_model.flow_logits.detach().cpu()},
            os.path.join(save_path, "warp_state.pt"),
        )
        attack_info = {
            "artifact_type": "patch_lattice_warp",
            "forget_class": self.forget_class,
            "attack_config": self.attack_config,
            "attack_history": self.attack_history,
        }
        if self.harp_state is not None:
            attack_info["harp_summary"] = self.harp_state.get("summary", {})
        with open(os.path.join(save_path, "attack_info.json"), "w", encoding="utf-8") as handle:
            json.dump(attack_info, handle, indent=2)

    def load_attack_artifact(self, load_path: str):
        state = torch.load(os.path.join(load_path, "warp_state.pt"), map_location=self.device)
        self.perturbed_target_model.flow_logits.data.copy_(state["flow_logits"].to(self.device))
        info_path = os.path.join(load_path, "attack_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as handle:
                info = json.load(handle)
            self.attack_history = info.get("attack_history", [])

    def get_attack_statistics(self) -> Dict[str, float]:
        flow = self.perturbed_target_model.get_flow_pixels().detach()
        flow_mag = torch.linalg.vector_norm(flow, dim=1)
        return {
            "image_size": int(self.perturbed_target_model.image_size),
            "patch_stride": int(self.perturbed_target_model.patch_stride),
            "flow_grid_size": int(self.perturbed_target_model.flow_grid_size),
            "max_flow_pixels": float(self.perturbed_target_model.max_flow_pixels),
            "flow_mean_abs": float(flow.abs().mean().item()),
            "flow_max_abs": float(flow.abs().max().item()),
            "flow_mag_mean": float(flow_mag.mean().item()),
            "flow_mag_max": float(flow_mag.max().item()),
        }
