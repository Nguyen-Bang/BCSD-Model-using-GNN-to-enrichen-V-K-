"""
Structured logging configuration for the BCSD pipeline.

This module provides centralized logging configuration per constitution
Principle III (Experiment Tracking) and Principle I (Research Documentation First).
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Configure structured logging for the application.
    
    Args:
        log_dir: Directory to store log files (default: "logs")
        log_level: Logging level (default: logging.INFO)
        log_to_file: Whether to log to file (default: True)
        log_to_console: Whether to log to console (default: True)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create log directory if it doesn't exist
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"bcsd_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("bcsd")
    logger.setLevel(log_level)
    logger.handlers = []  # Clear any existing handlers
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add file handler
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the module requesting the logger
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(f"bcsd.{name}")
