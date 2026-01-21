"""
Logger module for RAG services with debug mode control.
"""
import logging
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGLogger:
    """Custom logger for RAG services with debug mode control."""
    
    _loggers = {}
    
    @staticmethod
    def get_logger(name: str, debug_mode: Optional[bool] = None) -> logging.Logger:
        """
        Get or create a logger instance.
        
        Args:
            name: Name of the logger (usually __name__)
            debug_mode: Enable debug mode. If None, reads from DEBUG_MODE env var
            
        Returns:
            Configured logger instance
        """
        if name in RAGLogger._loggers:
            return RAGLogger._loggers[name]
        
        # Determine debug mode
        if debug_mode is None:
            debug_mode = os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes", "on")
        
        # Create logger
        logger = logging.getLogger(name)
        
        # Set level based on debug mode
        if debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers = []
        
        # Create console handler
        console_handler = logging.StreamHandler()
        
        if debug_mode:
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
        
        # Create formatter
        if debug_mode:
            # Detailed format for debug mode
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            # Simple format for normal mode
            formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        # Cache logger
        RAGLogger._loggers[name] = logger
        
        return logger


# Convenience function to get logger
def get_logger(name: str, debug_mode: Optional[bool] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Name of the logger (use __name__ for module-specific logging)
        debug_mode: Enable debug mode (default: from DEBUG_MODE env var)
        
    Returns:
        Configured logger instance
        
    Example:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("This is an info message")
        logger.debug("This is a debug message (only shown if DEBUG_MODE=true)")
    """
    return RAGLogger.get_logger(name, debug_mode)


# Example usage
if __name__ == "__main__":
    # Test with debug mode on
    print("Testing with DEBUG_MODE=true")
    os.environ["DEBUG_MODE"] = "true"
    RAGLogger._loggers.clear()
    logger1 = get_logger(__name__)
    logger1.debug("This is a debug message")
    logger1.info("This is an info message")
    logger1.warning("This is a warning message")
    logger1.error("This is an error message")
    
    print("\n" + "="*60 + "\n")
    
    # Test with debug mode off
    print("Testing with DEBUG_MODE=false")
    os.environ["DEBUG_MODE"] = "false"
    RAGLogger._loggers.clear()
    logger2 = get_logger(__name__)
    logger2.debug("This debug message should NOT appear")
    logger2.info("This info message should appear")
    logger2.warning("This warning message should appear")
    logger2.error("This error message should appear")
