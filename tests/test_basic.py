"""
Basic tests to verify ForgetGate-V installation and core functionality.
Run with: pytest tests/test_basic.py
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that core modules can be imported"""
    from src.data import DataManager
    from src.models.vit import create_vit_model
    from src.utils import set_seed, load_config
    assert DataManager is not None
    assert create_vit_model is not None
    assert set_seed is not None
    assert load_config is not None


def test_torch_cuda():
    """Test PyTorch CUDA availability"""
    if torch.cuda.is_available():
        _ = torch.cuda.get_device_name(0)
    assert True


def test_config_loading():
    """Test that configuration files can be loaded"""
    from src.utils import load_config
    config = load_config("configs/data.yaml")
    assert "cifar10" in config


def test_data_manager():
    """Test DataManager initialization"""
    from src.data import DataManager
    dm = DataManager()
    assert hasattr(dm, 'config')


if __name__ == "__main__":
    print("ForgetGate-V Basic Installation Test")
    print("=" * 40)

    tests = [
        test_imports,
        test_torch_cuda,
        test_config_loading,
        test_data_manager
    ]

    for test in tests:
        test()
        print()

    print("[SUCCESS] Installation looks good! Ready to run experiments.")
