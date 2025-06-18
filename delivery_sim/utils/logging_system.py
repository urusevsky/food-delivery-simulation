import logging
import os
import sys
from datetime import datetime

# Define custom log levels for simulation specific needs
# Standard levels are: DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
SIMULATION_EVENT = 15  # Between DEBUG and INFO
VALIDATION = 35        # Between WARNING and ERROR

# Register our custom levels with the logging module
logging.addLevelName(SIMULATION_EVENT, "SIMULATION")
logging.addLevelName(VALIDATION, "VALIDATION")

# Create a logger instance
logger = logging.getLogger("delivery_sim")

# Add methods to logger for our custom levels
def simulation_event(self, message, *args, **kwargs):
    """Log a simulation event message (level=SIMULATION_EVENT)."""
    self.log(SIMULATION_EVENT, message, *args, **kwargs)

def validation(self, message, *args, **kwargs):
    """Log a validation message (level=VALIDATION)."""
    self.log(VALIDATION, message, *args, **kwargs)

# Add our custom methods to the logger class
logging.Logger.simulation_event = simulation_event
logging.Logger.validation = validation

def configure_logging(config=None):
    """
    Configure the logging system based on the provided configuration.
    
    Args:
        config: LoggingConfig object with logging settings
               If None, default settings are used
    """
    # Set up default values
    console_level = logging.INFO
    file_level = logging.DEBUG
    log_to_file = False
    log_dir = "logs"
    log_file = None
    component_levels = {}
    
    # Update from config if provided
    if config:
        console_level = config.console_level
        file_level = config.file_level
        log_to_file = config.log_to_file
        log_dir = config.log_dir
        log_file = config.log_file
        component_levels = config.component_levels or {}
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set the root logger level to the lowest level we'll use
    # This ensures all messages reach the handlers, where they'll be filtered
    logger.setLevel(min(console_level, file_level if log_to_file else 100))
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(levelname)-10s | %(name)-25s | %(message)s'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-10s | %(name)s | %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if enabled
    if log_to_file:
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Generate file path if not provided
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"simulation_{timestamp}.log")
        else:
            log_file = os.path.join(log_dir, log_file)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Log the start of a new session
        logger.info(f"Logging session started, file: {log_file}")
    
    # Configure component-specific levels
    for component, level in component_levels.items():
        configure_component_level(component, level)
    
    return logger

def configure_component_level(component_name, level):
    """
    Set the logging level for a specific component.
    
    Args:
        component_name: Name of the component (e.g., 'service.pairing')
        level: Logging level to set
    """
    component_logger = logging.getLogger(f"delivery_sim.{component_name}")
    component_logger.setLevel(level)

def get_logger(name=None):
    """
    Get a logger instance, optionally as a child of the main logger.
    
    Args:
        name: Optional name for the logger, used to create hierarchical loggers
              If None, returns the main delivery_sim logger
    
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"delivery_sim.{name}")
    return logger

# Define level mapping helpers for configuration
def get_level_from_name(level_name):
    """Convert a level name string to its numeric value."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "SIMULATION": SIMULATION_EVENT,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "VALIDATION": VALIDATION,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    return level_map.get(level_name.upper(), logging.INFO)

# Initialize with default settings
configure_logging()