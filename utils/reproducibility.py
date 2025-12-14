"""
Reproducibility utilities for deterministic training and testing.

This module provides functions to set random seeds across all relevant libraries
to ensure reproducible results per constitution Principle III (Experiment Tracking).
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducible results.
    
    Args:
        seed: Random seed value (default: 42)
        
    Note:
        This function sets:
        - Python's random module seed
        - NumPy's random seed
        - PyTorch's manual seed (CPU and CUDA)
        - CUDA deterministic mode for reproducible GPU operations
        
    Warning:
        Setting deterministic=True may reduce performance but ensures reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Ensure deterministic behavior on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the appropriate PyTorch device (CUDA if available, else CPU).
    
    Returns:
        torch.device: CUDA device if GPU is available, otherwise CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_environment_info() -> dict:
    """
    Log environment information for reproducibility documentation.
    
    Returns:
        dict: Environment information including PyTorch version, CUDA availability, etc.
    """
    env_info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return env_info
