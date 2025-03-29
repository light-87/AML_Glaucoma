"""
Enhanced Logging Module

Centralized logging configuration with structured error reporting.
"""

import logging
import sys
import os
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Define error codes for better tracking
ERROR_CODES = {
    'DATA_LOAD_ERROR': 1001,
    'DATA_CLEAN_ERROR': 1002,
    'PREPROCESSING_ERROR': 1003,
    'MODEL_ERROR': 1004,
    'TRAINING_ERROR': 1005,
    'EVALUATION_ERROR': 1006,
    'CONFIG_ERROR': 1007,
    'PATH_ERROR': 1008,
    'IO_ERROR': 1009,
    'UNKNOWN_ERROR': 9999
}

class EnhancedLogger:
    """Enhanced logger with structured error reporting and context tracking."""
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        """Initialize logger with name and optional log directory.
        
        Args:
            name: Logger name
            log_dir: Directory to store log files
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.context = {}
        
        # Configure console handler if not already added
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler if log_dir is provided
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(exist_ok=True, parents=True)
            log_file = log_dir / f"{name.replace('.', '_')}.log"
            file_handler = logging.FileHandler(str(log_file))
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def set_context(self, **kwargs) -> None:
        """Set context information for logging.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context information."""
        self.context = {}
    
    def info(self, msg: str, **kwargs) -> None:
        """Log info message with context.
        
        Args:
            msg: Message to log
            **kwargs: Additional context for this log entry
        """
        context = {**self.context, **kwargs}
        if context:
            msg = f"{msg} | Context: {context}"
        self.logger.info(msg)
    
    def warning(self, msg: str, **kwargs) -> None:
        """Log warning message with context.
        
        Args:
            msg: Message to log
            **kwargs: Additional context for this log entry
        """
        context = {**self.context, **kwargs}
        if context:
            msg = f"{msg} | Context: {context}"
        self.logger.warning(msg)
    
    def error(self, msg: str, error_code: int = ERROR_CODES['UNKNOWN_ERROR'], 
              exc_info: bool = False, **kwargs) -> None:
        """Log error message with error code and context.
        
        Args:
            msg: Error message
            error_code: Error code for tracking
            exc_info: Whether to include exception info
            **kwargs: Additional context for this log entry
        """
        context = {**self.context, **kwargs, 'error_code': error_code}
        error_msg = f"ERROR {error_code}: {msg} | Context: {context}"
        self.logger.error(error_msg, exc_info=exc_info)
    
    def exception(self, msg: str, error_code: int = ERROR_CODES['UNKNOWN_ERROR'], **kwargs) -> None:
        """Log exception with stack trace, error code, and context.
        
        Args:
            msg: Error message
            error_code: Error code for tracking
            **kwargs: Additional context for this log entry
        """
        context = {**self.context, **kwargs, 'error_code': error_code}
        error_msg = f"EXCEPTION {error_code}: {msg} | Context: {context}"
        self.logger.exception(error_msg)
    
    def critical(self, msg: str, error_code: int = ERROR_CODES['UNKNOWN_ERROR'], **kwargs) -> None:
        """Log critical error with error code and context.
        
        Args:
            msg: Critical error message
            error_code: Error code for tracking
            **kwargs: Additional context for this log entry
        """
        context = {**self.context, **kwargs, 'error_code': error_code}
        error_msg = f"CRITICAL {error_code}: {msg} | Context: {context}"
        self.logger.critical(error_msg, exc_info=True)

def get_logger(name: str, log_dir: Optional[Union[str, Path]] = None) -> EnhancedLogger:
    """Get or create an enhanced logger with the given name.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
    
    Returns:
        Enhanced logger instance
    """
    if log_dir:
        log_dir = Path(log_dir)
    return EnhancedLogger(name, log_dir)

def log_exception_handler(func):
    """Decorator to catch and log exceptions.
    
    Args:
        func: Function to decorate
    
    Returns:
        Wrapped function with exception handling
    """
    def wrapper(*args, **kwargs):
        module_name = func.__module__
        func_name = func.__name__
        logger = get_logger(f"{module_name}.{func_name}")
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Determine error code based on module
            if 'data_loader' in module_name:
                error_code = ERROR_CODES['DATA_LOAD_ERROR']
            elif 'data_cleaner' in module_name:
                error_code = ERROR_CODES['DATA_CLEAN_ERROR']
            elif 'preprocessor' in module_name:
                error_code = ERROR_CODES['PREPROCESSING_ERROR']
            elif 'model' in module_name:
                error_code = ERROR_CODES['MODEL_ERROR']
            elif 'trainer' in module_name:
                error_code = ERROR_CODES['TRAINING_ERROR']
            elif 'evaluator' in module_name:
                error_code = ERROR_CODES['EVALUATION_ERROR']
            else:
                error_code = ERROR_CODES['UNKNOWN_ERROR']
            
            # Log the exception with full context
            logger.error(
                f"Error in {func_name}: {error_type} - {error_msg}",
                error_code=error_code,
                function=func_name,
                exc_info=True
            )
            
            # Re-raise the exception
            raise
    
    return wrapper