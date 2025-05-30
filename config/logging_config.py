# ============================================================================
# FILE: config/logging_config.py
# ============================================================================
import logging
import sys
from pathlib import Path

def setup_logging(log_level="INFO", log_file=None):
    """Setup logging configuration."""
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)
