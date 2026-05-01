"""
Universal repeated-tile attack for ViT patch embedding steering.
"""

import json
import os
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from ..models.normalize import create_imagenet_normalizer
from .fourier_perturbation import UniversalFourierAttack
from .universal_patch import _as_model_list


class UniversalTileModel(nn.Module):
    """Apply a low-amplitude tile repeated across the whole image."""

    def __init__(self, base_model: nn.Module, attack_config: Dict):
        super().__init__()
        self.base_model = base_model
        self.attack_config = dict(attack_config)
        self.tile_size = int(self.attack_config.get("tile_size", self.attack_config.get("patch_stride", 16)))
        self.epsilon = float(self.attack_config.get("epsilon", 8.0 / 255.0))
        self.spatial_gain = float(self.attack_config.get("spatial_gain", 12.0))
        self.phase_jitter = int(self.attack_config.get("phase_jitter", self.tile_size - 1))
        self.phase_grid = int(self.attack_config.get("phase_grid", 1))
        self.num_tiles = int(self.attack_config.get("num_tiles", self.phase_grid * self.phase_grid))
        self.eval_phase_offset_y: Optional[int] = None
        self.eval_phase_offset_x: Optional[int] = None

        init_scale = float(self.attack_config.get("init_scale", 1e-3))
        device = next(base_model.parameters()).device
        self.spectral_real = nn.Parameter(torch.zeros(self.num_tiles, 3, self.tile_size, self.tile_size, device=device))
        self.spectral_imag = nn.Parameter(torch.zeros(self.num_tiles, 3, self.tile_size, self.tile_size, device=device))
        nn.init.normal_(self.spectral_real, mean=0.0, std=init_scale)
        nn.init.normal_(self.spectral_imag, mean=0.0, std=init_scale)

    def get_tile(self) -> torch.Tensor:
        tile = torch.fft.ifft2(
            torch.complex(self.spectral_real, self.spectral_imag), norm="ortho"
        ).real
        tile = tile - tile.mean(dim=(2, 3), keepdim=True)
        return self.epsilon * torch.tanh(self.spatial_gain * tile)

    def get_delta(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        phase_offset_y: int = 0,
        phase_offset_x: int = 0,
    ) -> torch.Tensor:
        height = int(height or self.attack_config.get("image_size", 224))
        width = int(width or self.attack_config.get("image_size", 224))
        tile = self.get_tile()
        reps_y = (height + self.tile_size - 1) // self.tile_size
        reps_x = (width + self.tile_size - 1) // self.tile_size
        delta = torch.zeros(1, 3, height, width, device=tile.device, dtype=tile.dtype)
        for patch_y in range(reps_y):
            for patch_x in range(reps_x):
                top = patch_y * self.tile_size
                left = patch_x * self.tile_size
                bottom = min(height, top + self.tile_size)
                right = min(width, left + self.tile_size)
                phase_row = (patch_y + phase_offset_y) % self.phase_grid
                phase_col = (patch_x + phase_offset_x) % self.phase_grid
                phase_idx = (phase_row * self.phase_grid + phase_col) % self.num_tiles
                tile_view = tile[phase_idx : phase_idx + 1, :, : bottom - top, : right - left]
                delta[:, :, top:bottom, left:right] = tile_view
        return delta

    def get_complex_spectrum(self) -> torch.Tensor:
        delta = self.get_delta()
        return torch.fft.rfft2(delta, norm="ortho")

    def apply_perturbation(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = inputs.shape
        if self.phase_grid > 1:
            forced_offset_y = self.eval_phase_offset_y
            forced_offset_x = self.eval_phase_offset_x
            if forced_offset_y is not None and forced_offset_x is not None:
                delta = self.get_delta(height, width, int(forced_offset_y), int(forced_offset_x)).to(
                    device=inputs.device, dtype=inputs.dtype
                )
                delta_batch = delta.expand(batch_size, -1, -1, -1)
            else:
                deltas = []
                for _ in range(batch_size):
                    offset_y = int(torch.randint(0, self.phase_grid, (1,), device=inputs.device).item()) if self.training else 0
                    offset_x = int(torch.randint(0, self.phase_grid, (1,), device=inputs.device).item()) if self.training else 0
                    delta = self.get_delta(height, width, offset_y, offset_x).to(device=inputs.device, dtype=inputs.dtype)
                    deltas.append(delta)
                delta_batch = torch.cat(deltas, dim=0)
        else:
            delta = self.get_delta(height, width).to(device=inputs.device, dtype=inputs.dtype)
            if self.training and self.phase_jitter > 0:
                deltas = []
                for _ in range(batch_size):
                    shift_y = int(torch.randint(0, self.phase_jitter + 1, (1,), device=inputs.device).item())
                    shift_x = int(torch.randint(0, self.phase_jitter + 1, (1,), device=inputs.device).item())
                    deltas.append(torch.roll(delta, shifts=(shift_y, shift_x), dims=(2, 3)))
                delta_batch = torch.cat(deltas, dim=0)
            else:
                delta_batch = delta.expand(batch_size, -1, -1, -1)
        return torch.clamp(inputs + delta_batch, 0.0, 1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base_model(self.apply_perturbation(inputs))


class UniversalTileAttack(UniversalFourierAttack):
    """Reuse the Fourier attack training loop with a repeated-tile parameterization."""

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
        self.attack_history = []
        self.feature_mode = str(self.attack_config.get("feature_mode", "")).lower().strip()
        self.harp_state = None

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
            UniversalTileModel(model, self.attack_config).to(device)
            for model in self.target_clean_models
        ]
        for extra_model in self.perturbed_target_models[1:]:
            extra_model.spectral_real = self.perturbed_target_models[0].spectral_real
            extra_model.spectral_imag = self.perturbed_target_models[0].spectral_imag

        self.perturbed_target_model = self.perturbed_target_models[0]

        self.oracle_perturbed_model = None
        if self.oracle_clean_model is not None:
            self.oracle_perturbed_model = UniversalTileModel(
                self.oracle_clean_model, self.attack_config
            ).to(device)
            self.oracle_perturbed_model.spectral_real = self.perturbed_target_model.spectral_real
            self.oracle_perturbed_model.spectral_imag = self.perturbed_target_model.spectral_imag

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

    def _set_eval_phase_offsets(self, offset_y: Optional[int], offset_x: Optional[int]) -> None:
        for model in self.perturbed_target_models:
            model.eval_phase_offset_y = offset_y
            model.eval_phase_offset_x = offset_x
        if self.oracle_perturbed_model is not None:
            self.oracle_perturbed_model.eval_phase_offset_y = offset_y
            self.oracle_perturbed_model.eval_phase_offset_x = offset_x

    def _evaluate_resurrection(self, forget_loader):
        phase_grid = int(self.perturbed_target_model.phase_grid)
        if phase_grid <= 1:
            return super()._evaluate_resurrection(forget_loader)

        best_metrics = None
        best_phase = (0, 0)
        best_acc = -1.0
        best_loss = float("inf")
        for phase_y in range(phase_grid):
            for phase_x in range(phase_grid):
                self._set_eval_phase_offsets(phase_y, phase_x)
                metrics = super()._evaluate_resurrection(forget_loader)
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
                    best_phase = (phase_y, phase_x)
                    best_metrics = dict(metrics)

        self._set_eval_phase_offsets(*best_phase)
        if best_metrics is None:
            best_metrics = super()._evaluate_resurrection(forget_loader)
        best_metrics["eval_phase_offset_y"] = float(best_phase[0])
        best_metrics["eval_phase_offset_x"] = float(best_phase[1])
        return best_metrics

    def save_attack_artifact(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            {
                "tile_real": self.perturbed_target_model.spectral_real.detach().cpu(),
                "tile_imag": self.perturbed_target_model.spectral_imag.detach().cpu(),
            },
            os.path.join(save_path, "tile_state.pt"),
        )
        attack_info = {
            "artifact_type": "universal_tile",
            "forget_class": self.forget_class,
            "attack_config": self.attack_config,
            "attack_history": self.attack_history,
        }
        if self.harp_state is not None:
            attack_info["harp_summary"] = self.harp_state.get("summary", {})
        with open(os.path.join(save_path, "attack_info.json"), "w", encoding="utf-8") as handle:
            json.dump(attack_info, handle, indent=2)

    def load_attack_artifact(self, load_path: str):
        state_path = os.path.join(load_path, "tile_state.pt")
        state = torch.load(state_path, map_location=self.device)
        self.perturbed_target_model.spectral_real.data.copy_(state["tile_real"].to(self.device))
        self.perturbed_target_model.spectral_imag.data.copy_(state["tile_imag"].to(self.device))

        info_path = os.path.join(load_path, "attack_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as handle:
                info = json.load(handle)
            self.attack_history = info.get("attack_history", [])

    def get_attack_statistics(self) -> Dict[str, float]:
        delta = self.perturbed_target_model.get_delta().detach()
        tile = self.perturbed_target_model.get_tile().detach()
        return {
            "epsilon": float(self.perturbed_target_model.epsilon),
            "tile_size": int(self.perturbed_target_model.tile_size),
            "phase_grid": int(self.perturbed_target_model.phase_grid),
            "num_tiles": int(self.perturbed_target_model.num_tiles),
            "tile_mean_abs": float(tile.abs().mean().item()),
            "tile_max_abs": float(tile.abs().max().item()),
            "delta_mean_abs": float(delta.abs().mean().item()),
            "delta_max_abs": float(delta.abs().max().item()),
            "delta_l2": float(torch.linalg.vector_norm(delta).item()),
        }
