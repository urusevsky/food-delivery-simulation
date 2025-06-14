# delivery_sim/simulation/configuration.py

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
                 immediate_assignment_threshold,   # priority score threshold
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
    def __init__(self, simulation_duration, num_replications, master_seed):
        self.simulation_duration = simulation_duration  # minutes
        self.num_replications = num_replications        # 1 for single replication
        self.master_seed = master_seed                  # base seed for all randomness
    

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

class ScoringConfig:
    """
    Configuration for the priority scoring system.
    
    Separates infrastructure reality from business policy as outlined
    in the Priority Scoring System design document.
    """
    
    def __init__(self, 
                 # Business policy parameters (constant across configurations)
                 max_distance_ratio_multiplier=2.0,
                 max_acceptable_wait=30.0,
                 max_orders_per_trip=2,
                 
                 # Strategic weights (business preferences)
                 weight_distance=1/3,
                 weight_throughput=1/3,
                 weight_fairness=1/3,
                 
                 # Typical distance calculation settings
                 typical_distance_samples=1000):
        
        # Validate weights sum to 1 (with tolerance for floating point precision)
        total_weight = weight_distance + weight_throughput + weight_fairness
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        # Business policy parameters
        self.max_distance_ratio_multiplier = max_distance_ratio_multiplier
        self.max_acceptable_wait = max_acceptable_wait
        self.max_orders_per_trip = max_orders_per_trip
        
        # Strategic weights
        self.weight_distance = weight_distance
        self.weight_throughput = weight_throughput
        self.weight_fairness = weight_fairness
        
        # Calculation settings
        self.typical_distance_samples = typical_distance_samples
    
    def __str__(self):
        """String representation for debugging and logging."""
        return (f"ScoringConfig("
                f"weights=({self.weight_distance:.3f},{self.weight_throughput:.3f},{self.weight_fairness:.3f}), "
                f"max_ratio={self.max_distance_ratio_multiplier}, "
                f"max_wait={self.max_acceptable_wait}min)")

class FlatConfig:
    """
    A wrapper class that presents a flat interface for configuration attributes.
    
    This allows services to access attributes without knowing which config 
    object they belong to, simplifying service implementation.
    """
    def __init__(self, structural_config, operational_config, experiment_config, 
                 logging_config, scoring_config):
        self.structural_config = structural_config
        self.operational_config = operational_config
        self.experiment_config = experiment_config
        self.logging_config = logging_config
        self.scoring_config = scoring_config
    
    def __getattr__(self, name):
        # Try to find the attribute in each config object
        for config in [self.operational_config, self.structural_config, 
                      self.experiment_config, self.logging_config, 
                      self.scoring_config]:
            if hasattr(config, name):
                return getattr(config, name)
        
        raise AttributeError(f"'FlatConfig' object has no attribute '{name}'")

class SimulationConfig:
    """Complete configuration for a simulation experiment."""
    
    def __init__(self, structural_config, operational_config, 
                 experiment_config, logging_config, scoring_config):
        self.structural_config = structural_config
        self.operational_config = operational_config
        self.experiment_config = experiment_config
        self.logging_config = logging_config
        self.scoring_config = scoring_config
        
        # Create flat config for convenient service access
        self.flat_config = FlatConfig(
            structural_config, operational_config, experiment_config,
            logging_config, scoring_config
        )
    
    def __str__(self):
        """String representation for debugging and logging."""
        return (f"SimulationConfig:\n"
                f"  {self.structural_config}\n"
                f"  {self.operational_config}\n"
                f"  {self.experiment_config}\n"
                f"  {self.logging_config}\n"
                f"  {self.scoring_config}")