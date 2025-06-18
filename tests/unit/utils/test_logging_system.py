# tests/unit/utils/test_logging_system.py
import pytest
import logging
import os
import io
import sys
import time
from unittest.mock import patch, MagicMock
from delivery_sim.utils.logging_system import (
    configure_logging, get_logger, configure_component_level,
    get_level_from_name, logger, SIMULATION_EVENT, VALIDATION
)

# Test Group 1: Custom log levels and methods
def test_custom_log_levels_exist():
    """Test that custom log levels are defined with expected values."""
    # ASSERT
    assert SIMULATION_EVENT == 15, "SIMULATION_EVENT should be defined as level 15"
    assert VALIDATION == 35, "VALIDATION should be defined as level 35"
    assert hasattr(logging.Logger, 'simulation_event'), "Logger class should have simulation_event method"
    assert hasattr(logging.Logger, 'validation'), "Logger class should have validation method"

def test_custom_methods_added_to_logger():
    """Test that custom methods are properly added to the Logger class."""
    # ARRANGE
    test_logger = logging.getLogger('test_custom_methods')
    
    # Clear any existing handlers
    test_logger.handlers.clear()
    
    # Create a stream handler with a formatter that includes the level name
    stream = io.StringIO()
    stream_handler = logging.StreamHandler(stream)
    formatter = logging.Formatter('%(levelname)-10s: %(message)s')
    stream_handler.setFormatter(formatter)
    
    test_logger.addHandler(stream_handler)
    test_logger.setLevel(1)  # Set very low to capture all messages
    
    # ACT
    test_logger.simulation_event("Test simulation event")
    test_logger.validation("Test validation")
    
    # ASSERT
    output = stream.getvalue()
    assert "SIMULATION" in output, "simulation_event method should log with SIMULATION level"
    assert "VALIDATION" in output, "validation method should log with VALIDATION level"

# Test Group 2: Level name conversion
def test_get_level_from_name_standard_levels():
    """Test conversion of standard level names to numeric values."""
    # ACT & ASSERT
    assert get_level_from_name("DEBUG") == logging.DEBUG
    assert get_level_from_name("INFO") == logging.INFO
    assert get_level_from_name("WARNING") == logging.WARNING
    assert get_level_from_name("ERROR") == logging.ERROR
    assert get_level_from_name("CRITICAL") == logging.CRITICAL

def test_get_level_from_name_custom_levels():
    """Test conversion of custom level names to numeric values."""
    # ACT & ASSERT
    assert get_level_from_name("SIMULATION") == SIMULATION_EVENT
    assert get_level_from_name("VALIDATION") == VALIDATION

def test_get_level_from_name_case_insensitive():
    """Test that level name conversion is case-insensitive."""
    # ACT & ASSERT
    assert get_level_from_name("debug") == logging.DEBUG
    assert get_level_from_name("Info") == logging.INFO
    assert get_level_from_name("simulation") == SIMULATION_EVENT

def test_get_level_from_name_unknown():
    """Test that unknown level names return a default value (INFO)."""
    # ACT & ASSERT
    assert get_level_from_name("UNKNOWN_LEVEL") == logging.INFO
    assert get_level_from_name("") == logging.INFO

# Test Group 3: Core logger instance
def test_root_logger_exists():
    """Test that the main logger instance exists and has the correct name."""
    # ASSERT
    assert logger.name == "delivery_sim", "Root logger should be named 'delivery_sim'"

# Test Group 4: Logger retrieval
def test_get_logger_without_name():
    """Test that get_logger() without a name returns the main logger."""
    # ACT
    result = get_logger()
    
    # ASSERT
    assert result is logger, "get_logger() should return the main logger"
    assert result.name == "delivery_sim", "Main logger should be named 'delivery_sim'"

def test_get_logger_with_name():
    """Test that get_logger(name) returns a child logger with the correct name."""
    # ACT
    result = get_logger("test_component")
    
    # ASSERT
    assert result.name == "delivery_sim.test_component", "Child logger should have correctly formatted name"
    assert result is not logger, "Child logger should be a different instance than the main logger"

def test_get_logger_with_hierarchy():
    """Test that get_logger handles hierarchical component names."""
    # ACT
    parent = get_logger("parent")
    child = get_logger("parent.child")
    
    # ASSERT
    assert parent.name == "delivery_sim.parent"
    assert child.name == "delivery_sim.parent.child"

# Test Group 5: Basic configuration
@pytest.fixture
def reset_logger():
    """Fixture to reset logger state before and after tests."""
    # Store original handlers
    original_handlers = list(logger.handlers)
    original_level = logger.level
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    yield  # Run the test
    
    # Close and remove any handlers added during the test
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    
    # Restore original state
    for handler in original_handlers:
        logger.addHandler(handler)
    logger.setLevel(original_level)

def test_configure_logging_basic(reset_logger):
    """Test that basic logging configuration creates appropriate handlers."""
    # ARRANGE - Create a basic configuration (console only)
    from delivery_sim.simulation.configuration import LoggingConfig
    config = LoggingConfig(
        console_level="INFO",
        file_level="DEBUG",
        log_to_file=False  # Console only
    )
    
    # ACT
    configure_logging(config)
    
    # ASSERT
    assert len(logger.handlers) == 1, "Should create exactly one handler (console)"
    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler), "Handler should be a StreamHandler"
    assert handler.level == logging.INFO, "Console handler should have INFO level"

@patch('sys.stdout', new_callable=io.StringIO)
def test_log_message_appears_on_console(mock_stdout, reset_logger):
    """Test that log messages appear on the console with logger name for component identification."""
    # ARRANGE - Configure for console logging only
    from delivery_sim.simulation.configuration import LoggingConfig
    config = LoggingConfig(
        console_level="INFO",
        file_level="DEBUG",
        log_to_file=False
    )
    configure_logging(config)
    
    # ACT - Log a message
    logger.info("Test console message")
    
    # ASSERT - Focus on essential elements rather than exact format
    output = mock_stdout.getvalue()
    
    # Essential elements that should be present
    assert "INFO" in output, "Console should show log level"
    assert "delivery_sim" in output, "Console should show logger name for component identification"
    assert "Test console message" in output, "Console should show the actual message"
    
    # Verify it's a single line with newline
    lines = output.strip().split('\n')
    assert len(lines) == 1, "Should produce exactly one line of output"
    
    # The key improvement: logger name helps identify message source
    assert "delivery_sim" in lines[0], "Logger name should help identify message source"

# Test Group 6: File logging
@pytest.fixture
def temp_log_dir():
    """Fixture to create and clean up a temporary log directory."""
    import tempfile
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="test_logs_")
    
    yield temp_dir
    
    # Close all file handlers before cleanup
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
    
    # Give a small delay to ensure file handles are released
    time.sleep(0.1)
    
    # Clean up - remove all files and directory
    for filename in os.listdir(temp_dir):
        try:
            os.remove(os.path.join(temp_dir, filename))
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not remove file {filename}: {e}")
    
    try:
        os.rmdir(temp_dir)
    except (PermissionError, OSError) as e:
        print(f"Warning: Could not remove directory {temp_dir}: {e}")

def test_configure_logging_with_file(reset_logger, temp_log_dir):
    """Test that logging to file creates the expected files and handlers."""
    # ARRANGE
    from delivery_sim.simulation.configuration import LoggingConfig
    config = LoggingConfig(
        console_level="INFO",
        file_level="DEBUG",
        log_to_file=True,
        log_dir=temp_log_dir,
        log_file="test_log.log"
    )
    
    # ACT
    configure_logging(config)
    
    # ASSERT
    assert len(logger.handlers) == 2, "Should create two handlers (console and file)"
    
    # Find the file handler
    file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
            break
    
    assert file_handler is not None, "Should have a FileHandler"
    assert file_handler.level == logging.DEBUG, "File handler should have DEBUG level"
    assert os.path.exists(os.path.join(temp_log_dir, "test_log.log")), "Log file should be created"

def test_file_logging_writes_to_file(reset_logger, temp_log_dir):
    """Test that log messages are written to the log file."""
    # ARRANGE
    from delivery_sim.simulation.configuration import LoggingConfig
    config = LoggingConfig(
        console_level="INFO",
        file_level="DEBUG",
        log_to_file=True,
        log_dir=temp_log_dir,
        log_file="test_log.log"
    )
    configure_logging(config)
    
    # ACT
    logger.debug("Test debug message")
    logger.info("Test info message")
    
    # Force handlers to flush their buffers
    for handler in logger.handlers:
        handler.flush()
    
    # ASSERT
    log_file_path = os.path.join(temp_log_dir, "test_log.log")
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    assert "Test debug message" in log_content, "Debug message should be in log file"
    assert "Test info message" in log_content, "Info message should be in log file"

# Test Group 7: Component-specific logging
def test_configure_component_level(reset_logger):
    """Test that component-specific log levels can be configured."""
    # ARRANGE
    from delivery_sim.simulation.configuration import LoggingConfig
    config = LoggingConfig(
        console_level="INFO",
        file_level="DEBUG",
        log_to_file=False
    )
    configure_logging(config)
    
    # Get component loggers
    component_logger = get_logger("test_component")
    
    # ACT
    configure_component_level("test_component", logging.DEBUG)
    
    # ASSERT
    assert component_logger.level == logging.DEBUG, "Component logger should have DEBUG level"
    
    # The parent logger should still have its original level
    assert logger.level <= logging.INFO, "Main logger level should not be changed"

def test_component_levels_from_config(reset_logger):
    """Test that component levels specified in config are applied."""
    # ARRANGE
    from delivery_sim.simulation.configuration import LoggingConfig
    config = LoggingConfig(
        console_level="INFO",
        file_level="DEBUG",
        log_to_file=False,
        component_levels={
            "service.pairing": "DEBUG",
            "service.assignment": "WARNING"
        }
    )
    
    # ACT
    configure_logging(config)
    
    # ASSERT
    pairing_logger = get_logger("service.pairing")
    assignment_logger = get_logger("service.assignment")
    
    assert pairing_logger.level == logging.DEBUG, "Pairing logger should have DEBUG level"
    assert assignment_logger.level == logging.WARNING, "Assignment logger should have WARNING level"

def test_component_hierarchy_order_independence(reset_logger):
    """Test that parent/child logger configuration order doesn't affect final levels."""
    from delivery_sim.simulation.configuration import LoggingConfig
    
    # ARRANGE - Test configuration with child first, parent second
    config_child_first = LoggingConfig(
        console_level="DEBUG",
        file_level="DEBUG", 
        log_to_file=False,
        component_levels={
            "test.child": "DEBUG",        # Child first
            "test": "ERROR",              # Parent second
        }
    )
    
    # ACT - Apply child-first configuration
    configure_logging(config_child_first)
    
    # Get loggers after child-first configuration
    child_logger_1 = get_logger("test.child")
    parent_logger_1 = get_logger("test")
    
    # Store levels from child-first configuration
    child_level_1 = child_logger_1.level
    parent_level_1 = parent_logger_1.level
    
    # ARRANGE - Test configuration with parent first, child second  
    config_parent_first = LoggingConfig(
        console_level="DEBUG",
        file_level="DEBUG",
        log_to_file=False,
        component_levels={
            "test": "ERROR",              # Parent first
            "test.child": "DEBUG",        # Child second
        }
    )
    
    # ACT - Apply parent-first configuration
    configure_logging(config_parent_first)
    
    # Get loggers after parent-first configuration
    child_logger_2 = get_logger("test.child")
    parent_logger_2 = get_logger("test")
    
    # Store levels from parent-first configuration  
    child_level_2 = child_logger_2.level
    parent_level_2 = parent_logger_2.level
    
    # ASSERT - Both configurations should produce identical results
    assert child_level_1 == logging.DEBUG, "Child logger should have DEBUG level (child-first config)"
    assert parent_level_1 == logging.ERROR, "Parent logger should have ERROR level (child-first config)"
    assert child_level_2 == logging.DEBUG, "Child logger should have DEBUG level (parent-first config)"  
    assert parent_level_2 == logging.ERROR, "Parent logger should have ERROR level (parent-first config)"
    
    # The critical assertion: order should not matter
    assert child_level_1 == child_level_2, "Child logger level should be identical regardless of configuration order"
    assert parent_level_1 == parent_level_2, "Parent logger level should be identical regardless of configuration order"

def test_component_hierarchy_child_overrides_parent(reset_logger):
    """Test that explicit child logger levels override parent logger levels."""
    from delivery_sim.simulation.configuration import LoggingConfig
    
    # ARRANGE - Configuration where parent and child have conflicting levels
    config = LoggingConfig(
        console_level="DEBUG",
        file_level="DEBUG",
        log_to_file=False,
        component_levels={
            "entities": "ERROR",                    # Parent: suppress everything
            "entities.order": "DEBUG",              # Child: show debug messages  
            "entities.driver": "WARNING",           # Another child: show warnings+
            "services": "CRITICAL",                 # Another parent: almost nothing
            "services.order_arrival": "INFO",       # Child: show info+
        }
    )
    
    # ACT
    configure_logging(config)
    
    # Get all the loggers
    entities_logger = get_logger("entities")
    order_logger = get_logger("entities.order") 
    driver_logger = get_logger("entities.driver")
    services_logger = get_logger("services")
    arrival_logger = get_logger("services.order_arrival")
    
    # ASSERT - Each logger should have its explicitly configured level
    assert entities_logger.level == logging.ERROR, "entities logger should be ERROR"
    assert order_logger.level == logging.DEBUG, "entities.order should override parent to DEBUG" 
    assert driver_logger.level == logging.WARNING, "entities.driver should override parent to WARNING"
    assert services_logger.level == logging.CRITICAL, "services logger should be CRITICAL"
    assert arrival_logger.level == logging.INFO, "services.order_arrival should override parent to INFO"
    
    # ASSERT - Child levels should not be affected by their parents
    assert order_logger.level != entities_logger.level, "Child should not inherit parent level when explicitly set"
    assert arrival_logger.level != services_logger.level, "Child should not inherit parent level when explicitly set"

# Test Group 8: Default configuration
def test_configure_logging_without_config(reset_logger):
    """Test that configure_logging works with default values."""
    # ACT
    configure_logging()  # No config provided
    
    # ASSERT
    assert len(logger.handlers) > 0, "Should create at least one handler"
    assert logger.handlers[0].level == logging.INFO, "Default console level should be INFO"

# Test Group 9: Edge cases
def test_configure_logging_twice(reset_logger):
    """Test that configuring logging twice works correctly."""
    # ARRANGE
    from delivery_sim.simulation.configuration import LoggingConfig
    config1 = LoggingConfig(
        console_level="WARNING",
        log_to_file=False
    )
    
    # ACT
    configure_logging(config1)  # First configuration
    
    # Store handler count
    first_handler_count = len(logger.handlers)
    
    config2 = LoggingConfig(
        console_level="DEBUG",
        log_to_file=False
    )
    configure_logging(config2)  # Second configuration
    
    # ASSERT
    assert len(logger.handlers) == first_handler_count, "Should not add duplicate handlers"
    assert logger.handlers[0].level == logging.DEBUG, "Should update handler level"