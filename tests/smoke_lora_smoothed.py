"""Smoke: LoRA wrapping + smoothed objective forward+backward on the real model."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.models.normalize import create_imagenet_normalizer
from src.models.peft_lora import apply_lora_to_model, create_lora_config
from src.models.vit import create_vit_model
from src.unlearning.objectives import SmoothedMarginUnlearning
from src.utils import load_config


def main():
    cfg = load_config("configs/model.yaml")
    vit_cfg = dict(cfg["vit"]["tiny"])
    vit_cfg["pretrained"] = False
    base = create_vit_model(vit_cfg, num_classes=10)

    lora_cfg = create_lora_config(
        r=8, target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
    )
    model = apply_lora_to_model(base, lora_cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA wraps ok | trainable={n_train:,} / total={n_total:,}")

    normalizer = create_imagenet_normalizer()

    class Wrap(nn.Module):
        def __init__(self, m, n):
            super().__init__(); self.m, self.n = m, n
        def forward(self, x):
            return self.m(self.n(x))
    wrapped = Wrap(model, normalizer)

    obj = SmoothedMarginUnlearning(
        forget_class=0, num_classes=10, sigma=0.1, n_noise=2, gamma=1.0,
    )
    obj.set_model(wrapped)

    x = torch.rand(4, 3, 224, 224)
    y = torch.tensor([0, 0, 1, 2])
    logits = wrapped(x)
    loss = obj(logits, y, inputs=x)
    print(f"loss = {loss.item():.4f}")
    loss.backward()
    grad_norm_total = sum(
        (p.grad.norm().item() ** 2) for p in model.parameters() if p.grad is not None
    ) ** 0.5
    print(f"grad norm = {grad_norm_total:.4f}")
    assert grad_norm_total > 0


if __name__ == "__main__":
    main()
