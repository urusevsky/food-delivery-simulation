# delivery_sim/simulation/rng_manager.py
"""
Simplified RNG Managers for Clean Seed Management

Both managers take the same master_seed and handle their own seed transformations
for mathematical independence while maintaining semantic consistency.
"""

import numpy as np


class StructuralRNGManager:
    """
    Manages random number generation for structural elements like restaurant locations.
    
    These elements are fixed infrastructure that remain constant across replications
    but may vary between different structural configurations.
    """
    
    def __init__(self, master_seed):
        """
        Initialize structural RNG with master seed.
        
        Args:
            master_seed: Base seed for all randomness in the experiment
        """
        self.master_seed = master_seed
        self.rng = np.random.RandomState(master_seed)


class OperationalRNGManager:
    """
    Manages random number generation for operational processes.
    
    Maintains separate random streams for different process types to ensure
    independence while enabling Common Random Numbers (CRN) across configurations.
    """
    
    def __init__(self, master_seed, replication_number=0):
        """
        Initialize operational RNG with independent streams.
        
        Args:
            master_seed: Base seed for all randomness in the experiment
            replication_number: Replication index for stream separation
        """
        self.master_seed = master_seed
        self.replication_number = replication_number
        
        # Create independent streams for each random process
        # Uses large offsets to ensure mathematical independence
        base_offset = replication_number * 10000
        
        # Store both streams and their seeds for transparency
        self.stream_seeds = {
            'order_arrivals': master_seed + base_offset + 1,
            'driver_arrivals': master_seed + base_offset + 2,
            'service_duration': master_seed + base_offset + 3,
            'customer_locations': master_seed + base_offset + 4,
            'driver_initial_locations': master_seed + base_offset + 5,
            'restaurant_selection': master_seed + base_offset + 6
        }
        
        self.streams = {
            name: np.random.RandomState(seed) 
            for name, seed in self.stream_seeds.items()
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
            available_streams = ', '.join(self.streams.keys())
            raise ValueError(f"Unknown process: {process_name}. Available: {available_streams}")
        
        return self.streams[process_name]
    
    def get_sample_stream_seeds(self):
        """Return sample stream seeds to verify replication independence."""
        return {
            'order_arrivals': self.stream_seeds['order_arrivals'],
            'driver_arrivals': self.stream_seeds['driver_arrivals']
        }
    
    def get_all_stream_seeds(self):
        """Return all stream seeds for debugging purposes."""
        return self.stream_seeds.copy()
    
    def get_available_streams(self):
        """Get list of available stream names."""
        return list(self.streams.keys())