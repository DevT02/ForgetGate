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
    try:
        from src.data import DataManager
        from src.models.vit import create_vit_model
        from src.utils import set_seed, load_config
        print("[PASS] Core imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_torch_cuda():
    """Test PyTorch CUDA availability"""
    if torch.cuda.is_available():
        print(f"[PASS] CUDA available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("[PASS] Running on CPU (CUDA not available)")
        return True


def test_config_loading():
    """Test that configuration files can be loaded"""
    try:
        from src.utils import load_config
        config = load_config("configs/data.yaml")
        assert "cifar10" in config
        print("[PASS] Configuration loading works")
        return True
    except Exception as e:
        print(f"[FAIL] Config loading failed: {e}")
        return False


def test_data_manager():
    """Test DataManager initialization"""
    try:
        from src.data import DataManager
        dm = DataManager()
        assert hasattr(dm, 'config')
        print("[PASS] DataManager initialization works")
        return True
    except Exception as e:
        print(f"[FAIL] DataManager failed: {e}")
        return False


if __name__ == "__main__":
    print("ForgetGate-V Basic Installation Test")
    print("=" * 40)

    tests = [
        test_imports,
        test_torch_cuda,
        test_config_loading,
        test_data_manager
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("[SUCCESS] Installation looks good! Ready to run experiments.")
    else:
        print("[ERROR] Some tests failed. Check your installation.")
