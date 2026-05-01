"""
Conditional border-frame attack audit.

Learns image-conditioned frame content over a fixed border mask, then evaluates
whether a bounded frame blend can recover forget samples while minimally
affecting retain samples.
"""

import argparse
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.normalize import create_imagenet_normalizer
from src.utils import get_device, load_config, save_dict_to_json, set_seed


_AUDIT_PATH = os.path.join(os.path.dirname(__file__), "17_recovery_radius_audit.py")
_AUDIT_SPEC = importlib.util.spec_from_file_location("recovery_radius_audit", _AUDIT_PATH)
if _AUDIT_SPEC is None or _AUDIT_SPEC.loader is None:
    raise ImportError(f"Unable to load recovery audit helpers from {_AUDIT_PATH}")
_AUDIT = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT)

_PATCH_PATH = os.path.join(os.path.dirname(__file__), "20_conditional_patch_attack_audit.py")
_PATCH_SPEC = importlib.util.spec_from_file_location("conditional_patch_audit", _PATCH_PATH)
if _PATCH_SPEC is None or _PATCH_SPEC.loader is None:
    raise ImportError(f"Unable to load patch helpers from {_PATCH_PATH}")
_PATCH = importlib.util.module_from_spec(_PATCH_SPEC)
_PATCH_SPEC.loader.exec_module(_PATCH)

DEFAULT_MODEL_SUITES = list(getattr(_AUDIT, "DEFAULT_MODEL_SUITES"))
_load_model = getattr(_AUDIT, "_load_model")
_build_loader = getattr(_AUDIT, "_build_loader")
_predict = getattr(_AUDIT, "_predict")
_margin_objective = getattr(_AUDIT, "_margin_objective")
_suite_tag = getattr(_AUDIT, "_suite_tag")

_default_model_suites = getattr(_PATCH, "_default_model_suites")
_reshuffle_loader = getattr(_PATCH, "_reshuffle_loader")
_retain_target_calibration_loss = getattr(_PATCH, "_retain_target_calibration_loss")
_retain_clean_anchor_loss = getattr(_PATCH, "_retain_clean_anchor_loss")
_patch_area_penalty = getattr(_PATCH, "_patch_area_penalty")
_summarize_radius = getattr(_PATCH, "_summarize_radius")


class FrameAttackNet(nn.Module):
    def __init__(self, image_size: int, frame_width: int, width: int = 64):
        super().__init__()
        self.image_size = int(image_size)
        self.frame_width = int(frame_width)
        self.mask_template = self._build_mask_template()
        self.frame_area = int(self.mask_template.sum().item())

        self.backbone = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        hidden = width * 4 * 4
        self.frame_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 3 * self.frame_area),
        )

    def _build_mask_template(self) -> torch.Tensor:
        mask = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        w = self.frame_width
        mask[:, :w, :] = 1.0
        mask[:, -w:, :] = 1.0
        mask[:, :, :w] = 1.0
        mask[:, :, -w:] = 1.0
        return mask.bool()

    def forward(self, x_small: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x_small)
        frame_vals = self.frame_head(feat).view(-1, 3, self.frame_area)
        return torch.sigmoid(frame_vals)


def _frame_overlay(
    inputs: torch.Tensor,
    frame_vals: torch.Tensor,
    mask_template: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, _, image_size, _ = inputs.shape
    mask = mask_template.to(device=inputs.device).float().unsqueeze(0).expand(bsz, -1, -1, -1)
    canvas = inputs.clone()
    flat_mask = mask_template.view(-1)
    canvas_flat = canvas.view(bsz, 3, image_size * image_size)
    canvas_flat[:, :, flat_mask] = frame_vals
    canvas = canvas_flat.view_as(inputs)
    adv = torch.clamp(inputs + (alpha * mask * (canvas - inputs)), 0.0, 1.0)
    return adv, mask


def _train_frame_attack(
    attack_net: FrameAttackNet,
    model: nn.Module,
    normalizer: nn.Module,
    forget_loader,
    retain_loader,
    target_class: int,
    image_size: int,
    frame_width: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    retain_weight: float,
    retain_target_calib_weight: float,
    retain_target_calib_buffer: float,
    retain_clean_anchor_weight: float,
    retain_clean_anchor_buffer: float,
    retain_clean_anchor_temp: float,
    forget_ce_weight: float,
    forget_margin_weight: float,
    frame_area_weight: float,
) -> List[Dict[str, float]]:
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    optimizer = torch.optim.AdamW(attack_net.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[Dict[str, float]] = []
    retain_iter = iter(retain_loader)
    frame_fraction = float(attack_net.mask_template.float().mean().item())

    for epoch_idx in range(epochs):
        epoch_loss = 0.0
        epoch_forget_margin = 0.0
        epoch_retain_margin = 0.0
        epoch_area = 0.0
        n_steps = 0

        for forget_inputs, _ in forget_loader:
            forget_inputs = forget_inputs.to(next(attack_net.parameters()).device)
            try:
                retain_inputs, _ = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_inputs, _ = next(retain_iter)
            retain_inputs = retain_inputs.to(forget_inputs.device)

            x_small = F.interpolate(forget_inputs, size=(32, 32), mode="bilinear", align_corners=False)
            forget_frame = attack_net(x_small)
            forget_adv, forget_mask = _frame_overlay(forget_inputs, forget_frame, attack_net.mask_template, alpha=1.0)
            forget_logits = model(normalizer(forget_adv))
            forget_margin = _margin_objective(forget_logits, target_class)
            target_labels = torch.full((forget_inputs.size(0),), target_class, dtype=torch.long, device=forget_inputs.device)
            forget_ce = F.cross_entropy(forget_logits, target_labels)
            forget_loss = (forget_ce_weight * forget_ce) - (forget_margin_weight * forget_margin.mean())

            retain_small = F.interpolate(retain_inputs, size=(32, 32), mode="bilinear", align_corners=False)
            retain_frame = attack_net(retain_small)
            retain_adv, retain_mask = _frame_overlay(retain_inputs, retain_frame, attack_net.mask_template, alpha=1.0)
            with torch.no_grad():
                retain_clean_logits = model(normalizer(retain_inputs)).detach()
            retain_logits = model(normalizer(retain_adv))
            retain_margin = _margin_objective(retain_logits, target_class)
            retain_margin_mean = retain_margin.mean().detach()
            retain_loss = F.softplus(retain_margin).mean()
            if retain_target_calib_weight > 0.0:
                retain_loss = retain_loss + (
                    retain_target_calib_weight
                    * _retain_target_calibration_loss(
                        clean_logits=retain_clean_logits,
                        adv_logits=retain_logits,
                        target_class=target_class,
                        logit_buffer=retain_target_calib_buffer,
                    )
                )
            if retain_clean_anchor_weight > 0.0:
                retain_loss = retain_loss + (
                    retain_clean_anchor_weight
                    * _retain_clean_anchor_loss(
                        clean_logits=retain_clean_logits,
                        adv_logits=retain_logits,
                        target_class=target_class,
                        margin_buffer=retain_clean_anchor_buffer,
                        kl_temperature=retain_clean_anchor_temp,
                    )
                )

            area_penalty = 0.5 * (
                _patch_area_penalty(forget_mask, frame_fraction) + _patch_area_penalty(retain_mask, frame_fraction)
            )
            loss = forget_loss + (retain_weight * retain_loss) + (frame_area_weight * area_penalty)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(attack_net.parameters(), max_norm=5.0)
            optimizer.step()

            n_steps += 1
            epoch_loss += float(loss.item())
            epoch_forget_margin += float(forget_margin.mean().item())
            epoch_retain_margin += float(retain_margin_mean.item())
            epoch_area += float(forget_mask.mean().item())

        history.append(
            {
                "epoch": float(epoch_idx),
                "loss": epoch_loss / max(n_steps, 1),
                "forget_margin": epoch_forget_margin / max(n_steps, 1),
                "retain_margin": epoch_retain_margin / max(n_steps, 1),
                "mask_area_mean": epoch_area / max(n_steps, 1),
            }
        )
        print(
            f"Epoch {epoch_idx}: "
            f"loss={history[-1]['loss']:.4f} "
            f"forget_margin={history[-1]['forget_margin']:.4f} "
            f"retain_margin={history[-1]['retain_margin']:.4f} "
            f"mask_area={history[-1]['mask_area_mean']:.4f}"
        )

    attack_net.eval()
    return history


def _frame_recovery_search(
    model: nn.Module,
    normalizer: nn.Module,
    attack_net: FrameAttackNet,
    inputs: torch.Tensor,
    target_class: int,
    search_steps: int,
) -> Dict[str, Optional[float]]:
    clean_logits, clean_preds = _predict(model, normalizer, inputs)
    clean_target_prob = float(F.softmax(clean_logits, dim=1)[0, target_class].item())
    clean_pred_int = int(clean_preds.item())
    clean_success = bool((clean_preds == target_class).all().item())
    if clean_success:
        return {
            "success": True,
            "clean_success": True,
            "radius": 0.0,
            "clean_target_prob": clean_target_prob,
            "adv_target_prob": clean_target_prob,
            "clean_pred": clean_pred_int,
            "adv_pred": clean_pred_int,
        }

    with torch.no_grad():
        x_small = F.interpolate(inputs, size=(32, 32), mode="bilinear", align_corners=False)
        frame_vals = attack_net(x_small)

    low = 0.0
    high = None
    high_logits = None
    probe_schedule = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
    best_mask_area = None

    for alpha in probe_schedule:
        with torch.no_grad():
            adv_inputs, mask = _frame_overlay(inputs, frame_vals, attack_net.mask_template, alpha=float(alpha))
            adv_logits, adv_preds = _predict(model, normalizer, adv_inputs)
            if best_mask_area is None:
                best_mask_area = float(mask.mean().item())
        if bool((adv_preds == target_class).all().item()):
            high = float(alpha)
            high_logits = adv_logits
            break
        low = float(alpha)

    if high is None:
        with torch.no_grad():
            adv_inputs, mask = _frame_overlay(inputs, frame_vals, attack_net.mask_template, alpha=1.0)
            adv_logits, adv_preds = _predict(model, normalizer, adv_inputs)
            best_mask_area = float(mask.mean().item())
            if not bool((adv_preds == target_class).all().item()):
                adv_target_prob = float(F.softmax(adv_logits, dim=1)[0, target_class].item())
                return {
                    "success": False,
                    "clean_success": False,
                    "radius": None,
                    "clean_target_prob": clean_target_prob,
                    "adv_target_prob": adv_target_prob,
                    "clean_pred": clean_pred_int,
                    "adv_pred": int(adv_preds.item()),
                    "mask_area": best_mask_area,
                }
            high = 1.0
            high_logits = adv_logits

    for _ in range(search_steps):
        mid = 0.5 * (low + high)
        with torch.no_grad():
            adv_inputs, _mask = _frame_overlay(inputs, frame_vals, attack_net.mask_template, alpha=float(mid))
            adv_logits, adv_preds = _predict(model, normalizer, adv_inputs)
        if bool((adv_preds == target_class).all().item()):
            high = float(mid)
            high_logits = adv_logits
        else:
            low = float(mid)

    adv_probs = F.softmax(high_logits, dim=1)
    return {
        "success": True,
        "clean_success": False,
        "radius": float(high),
        "clean_target_prob": clean_target_prob,
        "adv_target_prob": float(adv_probs[0, target_class].item()),
        "clean_pred": clean_pred_int,
        "adv_pred": int(high_logits.argmax(dim=1).item()),
        "mask_area": best_mask_area,
    }


def main():
    parser = argparse.ArgumentParser(description="Conditional frame attack audit")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--eval-split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-suites", nargs="*", default=None)
    parser.add_argument("--max-train-forget-samples", type=int, default=512)
    parser.add_argument("--max-train-retain-samples", type=int, default=512)
    parser.add_argument("--max-eval-forget-samples", type=int, default=32)
    parser.add_argument("--max-eval-retain-samples", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--frame-width", type=int, default=16)
    parser.add_argument("--generator-width", type=int, default=64)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--retain-target-calib-weight", type=float, default=1.0)
    parser.add_argument("--retain-target-calib-buffer", type=float, default=0.0)
    parser.add_argument("--retain-clean-anchor-weight", type=float, default=1.0)
    parser.add_argument("--retain-clean-anchor-buffer", type=float, default=0.0)
    parser.add_argument("--retain-clean-anchor-temp", type=float, default=1.0)
    parser.add_argument("--forget-ce-weight", type=float, default=1.0)
    parser.add_argument("--forget-margin-weight", type=float, default=0.75)
    parser.add_argument("--frame-area-weight", type=float, default=10.0)
    parser.add_argument("--search-steps", type=int, default=6)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    experiment_suites = load_config(args.config)
    model_suites = args.model_suites or _default_model_suites(experiment_suites)
    if not model_suites:
        raise ValueError("No model suites provided and no default suites found in config")
    if int(args.eval_batch_size) != 1:
        raise ValueError("Frame audit is defined per sample. Use --eval-batch-size 1.")

    first_suite_cfg = experiment_suites[model_suites[0]]
    base_suite_name = first_suite_cfg["base_model_suite"]
    base_suite_cfg = experiment_suites[base_suite_name]
    dataset_name = base_suite_cfg["dataset"]
    model_type = base_suite_cfg["model"]
    forget_class = int(first_suite_cfg.get("unlearning", {}).get("forget_class", first_suite_cfg.get("forget_class", 0)))
    normalizer = create_imagenet_normalizer().to(device)
    image_size = 224

    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_type}")
    print(f"Forget class: {forget_class}")
    print(f"Model suites: {model_suites}")
    print(f"Frame width: {args.frame_width}")

    results = {
        "meta": {
            "seed": int(args.seed),
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "dataset": dataset_name,
            "model": model_type,
            "forget_class": int(forget_class),
            "model_suites": model_suites,
            "attack_family": "conditional_frame",
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "frame_width": int(args.frame_width),
            "generator_width": int(args.generator_width),
            "retain_weight": float(args.retain_weight),
            "retain_target_calib_weight": float(args.retain_target_calib_weight),
            "retain_clean_anchor_weight": float(args.retain_clean_anchor_weight),
            "frame_area_weight": float(args.frame_area_weight),
        },
        "forget_recovery": {},
        "retain_control": {},
        "training": {},
    }

    for suite_name in model_suites:
        print(f"\nTraining conditional frame attacker for {suite_name}...")
        model, suite_dataset, suite_model = _load_model(experiment_suites, suite_name, args.seed, device)
        if suite_dataset != dataset_name or suite_model != model_type:
            raise ValueError(f"Suite {suite_name} resolved to {suite_dataset}/{suite_model}, expected {dataset_name}/{model_type}")
        model.eval()

        train_forget_loader = _reshuffle_loader(
            _build_loader(dataset_name=dataset_name, split=args.train_split, batch_size=args.train_batch_size, num_workers=args.num_workers, forget_class=forget_class, retain=False, max_samples=args.max_train_forget_samples),
            batch_size=int(args.train_batch_size),
            num_workers=int(args.num_workers),
        )
        train_retain_loader = _reshuffle_loader(
            _build_loader(dataset_name=dataset_name, split=args.train_split, batch_size=args.train_batch_size, num_workers=args.num_workers, forget_class=forget_class, retain=True, max_samples=args.max_train_retain_samples),
            batch_size=int(args.train_batch_size),
            num_workers=int(args.num_workers),
        )
        eval_forget_loader = _build_loader(dataset_name=dataset_name, split=args.eval_split, batch_size=args.eval_batch_size, num_workers=args.num_workers, forget_class=forget_class, retain=False, max_samples=args.max_eval_forget_samples)
        eval_retain_loader = _build_loader(
            dataset_name=dataset_name,
            split=args.eval_split,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            forget_class=forget_class,
            retain=True,
            max_samples=args.max_eval_retain_samples,
        )

        attack_net = FrameAttackNet(image_size=image_size, frame_width=int(args.frame_width), width=int(args.generator_width)).to(device)

        history = _train_frame_attack(
            attack_net=attack_net,
            model=model,
            normalizer=normalizer,
            forget_loader=train_forget_loader,
            retain_loader=train_retain_loader,
            target_class=forget_class,
            image_size=image_size,
            frame_width=int(args.frame_width),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            retain_weight=float(args.retain_weight),
            retain_target_calib_weight=float(args.retain_target_calib_weight),
            retain_target_calib_buffer=float(args.retain_target_calib_buffer),
            retain_clean_anchor_weight=float(args.retain_clean_anchor_weight),
            retain_clean_anchor_buffer=float(args.retain_clean_anchor_buffer),
            retain_clean_anchor_temp=float(args.retain_clean_anchor_temp),
            forget_ce_weight=float(args.forget_ce_weight),
            forget_margin_weight=float(args.forget_margin_weight),
            frame_area_weight=float(args.frame_area_weight),
        )

        ckpt_dir = os.path.join("checkpoints", "conditional_attack")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"conditional_frame_{suite_name}_seed_{args.seed}.pt")
        torch.save({"state_dict": attack_net.state_dict(), "training_history": history, "meta": results["meta"]}, ckpt_path)

        print(f"Evaluating conditional frame attacker for {suite_name}...")
        forget_records: List[Dict[str, Optional[float]]] = []
        for idx, (inputs, labels) in enumerate(eval_forget_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            record = _frame_recovery_search(model=model, normalizer=normalizer, attack_net=attack_net, inputs=inputs, target_class=forget_class, search_steps=int(args.search_steps))
            record["sample_index"] = idx
            record["label"] = int(labels.item())
            record["clean_correct"] = float(record["clean_pred"] == int(labels.item()))
            record["adv_correct"] = float(record["adv_pred"] == int(labels.item()))
            forget_records.append(record)

        retain_records: List[Dict[str, Optional[float]]] = []
        for idx, (inputs, labels) in enumerate(eval_retain_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            record = _frame_recovery_search(model=model, normalizer=normalizer, attack_net=attack_net, inputs=inputs, target_class=forget_class, search_steps=int(args.search_steps))
            record["sample_index"] = idx
            record["label"] = int(labels.item())
            record["clean_correct"] = float(record["clean_pred"] == int(labels.item()))
            record["adv_correct"] = float(record["adv_pred"] == int(labels.item()))
            retain_records.append(record)

        results["training"][suite_name] = {"history": history, "checkpoint_path": ckpt_path}
        results["forget_recovery"][suite_name] = {"summary": _summarize_radius(forget_records), "per_sample": forget_records}
        results["retain_control"][suite_name] = {"summary": _summarize_radius(retain_records), "per_sample": retain_records}

    attack_tag = (
        f"conditional_frame_fw{int(args.frame_width)}"
        f"_rw{str(args.retain_weight).replace('.', 'p')}"
        f"_rtc{str(args.retain_target_calib_weight).replace('.', 'p')}"
        f"_rca{str(args.retain_clean_anchor_weight).replace('.', 'p')}"
    )
    save_path = (
        f"results/analysis/conditional_frame_{dataset_name}_{model_type}_forget{forget_class}"
        f"_seed_{args.seed}_{args.eval_split}_{attack_tag}_{_suite_tag(model_suites)}.json"
    )
    save_dict_to_json(results, save_path)
    print(f"\nSaved conditional frame audit to: {save_path}")

    print("\nForget recovery summary")
    print("Model                                   | success | median alpha | mean alpha | mask area | acc drop")
    print("-" * 110)
    for suite_name in model_suites:
        summary = results["forget_recovery"][suite_name]["summary"]
        print(f"{suite_name:<39} | {summary['success_rate']:.3f} | {summary['median_radius']:.5f} | {summary['mean_radius']:.5f} | {summary['mean_mask_area']:.4f} | {summary['accuracy_drop']:.3f}")

    print("\nRetain-to-forget control summary")
    print("Model                                   | success | median alpha | mean alpha | acc drop")
    print("-" * 102)
    for suite_name in model_suites:
        summary = results["retain_control"][suite_name]["summary"]
        print(f"{suite_name:<39} | {summary['success_rate']:.3f} | {summary['median_radius']:.5f} | {summary['mean_radius']:.5f} | {summary['accuracy_drop']:.3f}")


if __name__ == "__main__":
    main()
