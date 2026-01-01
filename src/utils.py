"""
Utility functions for ForgetGate-V
Seeding, checkpoint I/O, logging, etc.
"""

import torch
import numpy as np
import random
import os
import json
import yaml
import re
from pathlib import Path
from typing import Any, Dict, Optional


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


_NUM_RE = re.compile(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$')


def _coerce_numbers(obj):
    """Recursively coerce numeric strings to numbers"""
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if _NUM_RE.match(s):
            # prefer int when it looks like an int
            if '.' not in s and 'e' not in s.lower():
                try:
                    return int(s)
                except ValueError:
                    pass
            try:
                return float(s)
            except ValueError:
                return obj
    return obj


def load_config(config_path: str):
    """Load YAML config file and coerce numeric strings to numbers"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return _coerce_numbers(cfg)


def save_config(config, save_path: str):
    """Save config to YAML"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_experiment_name(suite_name: str, seed: int, **kwargs):
    """Generate experiment name from params"""
    name_parts = [suite_name, f"seed_{seed}"]

    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            name_parts.append(f"{key}_{value}")
        elif isinstance(value, str) and len(value) < 20:  # Don't make names too long
            name_parts.append(f"{key}_{value}")

    return "_".join(name_parts)


def ensure_dir(path: str):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_dict_to_json(data: Dict, save_path: str):
    """Save dictionary to JSON file"""
    ensure_dir(os.path.dirname(save_path))

    # Convert numpy types to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj

    serializable_data = {k: convert_to_serializable(v) for k, v in data.items()}

    with open(save_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def load_dict_from_json(load_path: str) -> Dict:
    """Load dictionary from JSON file"""
    with open(load_path, 'r') as f:
        data = json.load(f)
    return data


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_info(model: torch.nn.Module, model_name: str = ""):
    """Print model information"""
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB")
    print(f"{'='*50}\n")


def format_metrics(metrics: Dict[str, Any], precision: int = 4) -> str:
    """Format metrics dictionary as string"""
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                formatted.append(f"{key}: {value:.{precision}f}")
            else:
                formatted.append(f"{key}: {value}")
        else:
            formatted.append(f"{key}: {value}")

    return " | ".join(formatted)


def create_experiment_log(experiment_name: str, config: Dict,
                         log_dir: str = "results/logs") -> str:
    """Create experiment log file path"""
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, f"{experiment_name}.jsonl")
    return log_path


def log_experiment(log_path: str, data: Dict):
    """Log experiment data to JSONL file"""
    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj

    serializable_data = {k: make_serializable(v) for k, v in data.items()}

    with open(log_path, 'a') as f:
        json.dump(serializable_data, f)
        f.write('\n')


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get torch device"""
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_str)

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def time_function(func):
    """Decorator to time function execution"""
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressMeter:
    """Displays training progress"""

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def cleanup_checkpoints(checkpoint_dir: str, keep_last: int = 3):
    """Clean up old checkpoints, keeping only the most recent ones"""
    if not os.path.exists(checkpoint_dir):
        return

    # Get all checkpoint files
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt') or file.endswith('.pth'):
            filepath = os.path.join(checkpoint_dir, file)
            mtime = os.path.getmtime(filepath)
            checkpoints.append((filepath, mtime))

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x[1], reverse=True)

    # Remove old checkpoints
    for filepath, _ in checkpoints[keep_last:]:
        try:
            os.remove(filepath)
            print(f"Removed old checkpoint: {filepath}")
        except OSError:
            pass


if __name__ == "__main__":
    # Test utilities
    set_seed(42)

    # Test device
    device = get_device()
    print(f"Device: {device}")

    # Test config loading
    print("Utils module loaded successfully")
