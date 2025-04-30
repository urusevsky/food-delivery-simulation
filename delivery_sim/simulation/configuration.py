class StructuralConfig:
    """Configuration for fixed infrastructure elements of the simulation."""
    
    def __init__(self, 
                 delivery_area_size,  # km (square area)
                 num_restaurants,
                 restaurant_pattern,  # 'clustered', 'dispersed', 'mixed'
                 driver_speed):       # km per minute
        
        self.delivery_area_size = delivery_area_size
        self.num_restaurants = num_restaurants
        self.restaurant_pattern = restaurant_pattern
        self.driver_speed = driver_speed
        
    def __str__(self):
        """String representation for debugging and logging."""
        return (f"StructuralConfig(area_size={self.delivery_area_size}km, "
                f"restaurants={self.num_restaurants}, "
                f"pattern={self.restaurant_pattern})")


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
                 random_seed,                    # base seed for random number generation
                 metrics_collection_interval,    # minutes between metric snapshots
                 event_recording_enabled):       # whether to record detailed events
        
        self.simulation_duration = simulation_duration
        self.warmup_period = warmup_period
        self.num_replications = num_replications
        self.random_seed = random_seed
        self.metrics_collection_interval = metrics_collection_interval
        self.event_recording_enabled = event_recording_enabled
    
    def __str__(self):
        """String representation for debugging and logging."""
        return (f"ExperimentConfig("
                f"duration={self.simulation_duration}min, "
                f"warmup={self.warmup_period}min, "
                f"replications={self.num_replications})")


class SimulationConfig:
    """Complete configuration for a simulation experiment."""
    
    def __init__(self, structural_config, operational_config, experiment_config):
        self.structural_config = structural_config
        self.operational_config = operational_config
        self.experiment_config = experiment_config
    
    def __str__(self):
        """String representation for debugging and logging."""
        return (f"SimulationConfig:\n"
                f"  {self.structural_config}\n"
                f"  {self.operational_config}\n"
                f"  {self.experiment_config}")