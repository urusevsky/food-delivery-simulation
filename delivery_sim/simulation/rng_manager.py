import numpy as np

class StructuralRNGManager:
    """
    Manages random number generation for structural elements like restaurant locations.
    
    These elements are fixed infrastructure that remain constant across replications
    but may vary between different structural configurations.
    """
    def __init__(self, structural_seed):
        """
        Initialize the structural RNG manager with a seed.
        
        Args:
            structural_seed: Seed for structural random generation
        """
        self.structural_seed = structural_seed
        self.rng = np.random.RandomState(structural_seed)

class OperationalRNGManager:
    """
    Manages random number generation for operational processes.
    
    Maintains separate random streams for different process types to ensure
    independence while enabling Common Random Numbers (CRN) across configurations.
    """
    def __init__(self, base_seed, replication_number=0):
        """
        Initialize operational random streams for a specific replication.
        
        Args:
            base_seed: Base seed for the experiment
            replication_number: Current replication number (default: 0)
        """
        # Create independent streams for each random process
        self.streams = {
            'order_arrivals': np.random.RandomState(base_seed + replication_number * 10000 + 1),
            'driver_arrivals': np.random.RandomState(base_seed + replication_number * 10000 + 2),
            'service_duration': np.random.RandomState(base_seed + replication_number * 10000 + 3),
            'customer_locations': np.random.RandomState(base_seed + replication_number * 10000 + 4),
            'driver_initial_locations': np.random.RandomState(base_seed + replication_number * 10000 + 5),
            'restaurant_selection': np.random.RandomState(base_seed + replication_number * 10000 + 6)
        }
    
    def get_stream(self, process_name):
        """
        Get the random stream for a specific process.
        
        Args:
            process_name: Name of the random process
            
        Returns:
            RandomState: The process-specific random stream
            
        Raises:
            ValueError: If process name is unknown
        """
        if process_name not in self.streams:
            raise ValueError(f"Unknown process: {process_name}")
        return self.streams[process_name]