class StructuralConfig:
    """Configuration for fixed infrastructure elements of the simulation."""
    
def __init__(self, 
             delivery_area_size,  # km (square area)
             num_restaurants,
             driver_speed):       # km per minute
    
    self.delivery_area_size = delivery_area_size
    self.num_restaurants = num_restaurants
    self.driver_speed = driver_speed
        
    def __str__(self):
        """String representation for debugging and logging."""
        return (f"StructuralConfig(area_size={self.delivery_area_size}km, "
                f"restaurants={self.num_restaurants})")


class OperationalConfig:
    """Configuration for dynamic operational elements of the simulation."""
    
    def __init__(self, 
                 # Arrival intervals (minutes)
                 mean_order_inter_arrival_time,    # minutes between orders
                 mean_driver_inter_arrival_time,   # minutes between drivers
                 
                 # Pairing configuration
                 pairing_enabled,
                 restaurants_proximity_threshold,  # km
                 customers_proximity_threshold,    # km
                 
                 # Driver service configuration
                 mean_service_duration,            # minutes
                 service_duration_std_dev,         # minutes
                 min_service_duration,             # minutes
                 max_service_duration,             # minutes
                 
                 # Assignment parameters
                 throughput_factor,                # km per additional order
                 age_factor,                       # km per minute waiting
                 immediate_assignment_threshold,   # km (adjusted cost)
                 periodic_interval):               # minutes between optimizations

        # Arrival process parameters
        self.mean_order_inter_arrival_time = mean_order_inter_arrival_time
        self.mean_driver_inter_arrival_time = mean_driver_inter_arrival_time
        
        # Pairing configuration
        self.pairing_enabled = pairing_enabled
        self.restaurants_proximity_threshold = restaurants_proximity_threshold
        self.customers_proximity_threshold = customers_proximity_threshold
        
        # Driver service duration parameters
        self.mean_service_duration = mean_service_duration
        self.service_duration_std_dev = service_duration_std_dev
        self.min_service_duration = min_service_duration
        self.max_service_duration = max_service_duration
        
        # Assignment logic parameters
        self.throughput_factor = throughput_factor
        self.age_factor = age_factor
        self.immediate_assignment_threshold = immediate_assignment_threshold
        self.periodic_interval = periodic_interval
    
    def __str__(self):
        """String representation for debugging and logging."""
        return (f"OperationalConfig("
                f"order_interval={self.mean_order_inter_arrival_time}min, "
                f"driver_interval={self.mean_driver_inter_arrival_time}min, "
                f"pairing={'on' if self.pairing_enabled else 'off'}, "
                f"threshold={self.immediate_assignment_threshold})")


class ExperimentConfig:
    """Configuration for experimental parameters and data collection."""
    
    def __init__(self,
                 simulation_duration,            # minutes
                 warmup_period,                  # minutes to exclude from statistics
                 num_replications,               # number of independent simulation runs
                 master_seed,                    # primary seed controlling all randomness
                 metrics_collection_interval,    # minutes between metric snapshots
                 event_recording_enabled):       # whether to record detailed events
        
        self.simulation_duration = simulation_duration
        self.warmup_period = warmup_period
        self.num_replications = num_replications
        self.master_seed = master_seed  # Renamed from random_seed for clarity
        self.metrics_collection_interval = metrics_collection_interval
        self.event_recording_enabled = event_recording_enabled
    
    def generate_structural_seed(self):
        """
        Deterministically derive the structural seed from the master seed.
        
        This seed controls fixed infrastructure elements like restaurant locations
        that remain constant across replications but may vary between configurations.
        
        Returns:
            int: Derived structural seed
        """
        return self.master_seed * 17 + 31  # Simple but effective derivation
    
    def generate_operational_base_seed(self):
        """
        Deterministically derive the operational base seed from the master seed.
        
        This seed serves as the base for all operational random processes,
        which are further diversified by replication number.
        
        Returns:
            int: Derived operational base seed
        """
        return self.master_seed * 23 + 41  # Different formula from structural
    
    def __str__(self):
        """String representation for debugging and logging."""
        return (f"ExperimentConfig("
                f"duration={self.simulation_duration}min, "
                f"warmup={self.warmup_period}min, "
                f"replications={self.num_replications}, "
                f"master_seed={self.master_seed})")

class LoggingConfig:
    """Configuration for logging system behavior."""
    
    def __init__(self,
                 console_level="INFO",      # Level for console output
                 file_level="DEBUG",        # Level for file output
                 log_to_file=False,         # Whether to log to file
                 log_dir="logs",            # Directory for log files
                 log_file=None,             # Specific log file name (optional)
                 component_levels=None):    # Dict of component-specific levels
        
        # Convert string levels to numeric values if needed
        from delivery_sim.utils.logging_system import get_level_from_name
        self.console_level = (get_level_from_name(console_level) 
                             if isinstance(console_level, str) 
                             else console_level)
        self.file_level = (get_level_from_name(file_level) 
                          if isinstance(file_level, str) 
                          else file_level)
        self.log_to_file = log_to_file
        self.log_dir = log_dir
        self.log_file = log_file
        self.component_levels = component_levels or {}
        
        # If component_levels contains string levels, convert them
        if component_levels:
            for component, level in list(self.component_levels.items()):
                if isinstance(level, str):
                    self.component_levels[component] = get_level_from_name(level)

class SimulationConfig:
    """Complete configuration for a simulation experiment."""
    
    def __init__(self, structural_config, operational_config, 
                 experiment_config, logging_config=None):
        self.structural_config = structural_config
        self.operational_config = operational_config
        self.experiment_config = experiment_config
        self.logging_config = logging_config or LoggingConfig()
    
    def __str__(self):
        """String representation for debugging and logging."""
        return (f"SimulationConfig:\n"
                f"  {self.structural_config}\n"
                f"  {self.operational_config}\n"
                f"  {self.experiment_config}")