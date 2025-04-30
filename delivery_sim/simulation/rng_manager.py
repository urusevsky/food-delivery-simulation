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
        
    def generate_uniform(self, low, high, size=None):
        """
        Generate uniformly distributed random numbers.
        
        Args:
            low: Lower bound
            high: Upper bound
            size: Output shape (optional)
            
        Returns:
            Uniform random samples
        """
        return self.rng.uniform(low, high, size)
    
    def generate_normal(self, mean, std, size=None):
        """
        Generate normally distributed random numbers.
        
        Args:
            mean: Mean of distribution
            std: Standard deviation
            size: Output shape (optional)
            
        Returns:
            Normal random samples
        """
        return self.rng.normal(mean, std, size)
    
    def choice(self, items, size=None, replace=True, p=None):
        """
        Randomly select items from a list.
        
        Args:
            items: Array-like containing items to choose from
            size: Output shape (optional)
            replace: Whether to sample with replacement
            p: Probability weights for each item
            
        Returns:
            Selected items
        """
        return self.rng.choice(items, size=size, replace=replace, p=p)

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