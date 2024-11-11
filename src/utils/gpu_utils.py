"""Utility functions for GPU setup and management"""
import logging
import torch

logger = logging.getLogger(__name__)

def setup_gpu() -> torch.device:
    """
    Setup GPU device with proper error handling.
    
    Returns:
        torch.device: Device to use (cuda or cpu)
    """
    if torch.cuda.is_available():
        try:
            # Try to initialize CUDA
            device = torch.device('cuda')
            # Test CUDA initialization
            torch.zeros(1).cuda()
            
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
            logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f}GB memory")
            
            return device
            
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {str(e)}")
            logger.warning("Falling back to CPU")
            return torch.device('cpu')
    else:
        logger.warning("No CUDA-capable GPU found. Using CPU")
        return torch.device('cpu')

def get_available_memory() -> float:
    """
    Get available GPU memory in GB.
    
    Returns:
        float: Available memory in GB
    """
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        except:
            return 0.0
    return 0.0