"""Smoke test for SmoothedMarginUnlearning (Theorem 4 objective)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.unlearning.objectives import create_unlearning_objective, SmoothedMarginUnlearning


def main():
    obj = create_unlearning_objective(
        "smoothed_margin", forget_class=0, num_classes=10,
        sigma=0.1, n_noise=2, gamma=1.0,
    )
    assert isinstance(obj, SmoothedMarginUnlearning)

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 * 32 * 32, 10)
        def forward(self, x):
            return self.linear(x.flatten(1))

    net = TinyNet()
    obj.set_model(net)

    x = torch.rand(8, 3, 32, 32)
    labels = torch.tensor([0, 0, 0, 1, 2, 3, 4, 5])
    logits = net(x)
    loss = obj(logits, labels, inputs=x)
    assert loss.requires_grad
    loss.backward()
    g = net.linear.weight.grad.norm().item()
    print(f"smoothed_margin smoke: loss={loss.item():.4f} grad_norm={g:.4f}")

    # Edge case 1: no forget examples in batch
    labels2 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    net.zero_grad()
    loss2 = obj(net(x), labels2, inputs=x)
    loss2.backward()
    print(f"no-forget batch: loss={loss2.item():.4f}")

    # Edge case 2: all-forget batch
    labels3 = torch.zeros(8, dtype=torch.long)
    net.zero_grad()
    loss3 = obj(net(x), labels3, inputs=x)
    loss3.backward()
    print(f"all-forget batch: loss={loss3.item():.4f}")


if __name__ == "__main__":
    main()
