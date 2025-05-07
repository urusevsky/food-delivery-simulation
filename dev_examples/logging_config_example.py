# Creating a configuration with custom logging settings
logging_config = LoggingConfig(
    console_level="INFO",
    file_level="DEBUG", 
    log_to_file=True,
    component_levels={
        "service.pairing": "DEBUG",       # More verbose for pairing service
        "service.assignment": "INFO",     # Standard level for assignment
        "service.delivery": "WARNING"     # Only warnings for delivery service
    }
)

# Creating the full simulation configuration
sim_config = SimulationConfig(
    structural_config=structural_config,
    operational_config=operational_config,
    experiment_config=experiment_config,
    logging_config=logging_config
)

# In simulation_runner.py
def __init__(self, config):
    # Configure logging based on config
    from delivery_sim.utils.logging_system import configure_logging
    configure_logging(config.logging_config)
    
    # Get a logger for this component
    self.logger = get_logger("simulation.runner")