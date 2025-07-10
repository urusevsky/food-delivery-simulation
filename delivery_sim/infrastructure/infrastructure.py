# delivery_sim/infrastructure/infrastructure.py
"""
Infrastructure: First-Class Entity for Food Delivery Simulation

This module implements Infrastructure as an independent, reusable entity that
encapsulates the complete physical delivery environment. Infrastructure instances
can be shared across multiple experimental configurations for efficiency and
analytical consistency.

Key Design Principles:
- Self-contained: Owns all physical environment components
- Deterministic: Same structural_config + master_seed = identical infrastructure  
- Reusable: Can be shared across multiple operational configurations
- Analysis-ready: Provides access for comprehensive infrastructure analysis
"""

from delivery_sim.entities.restaurant import Restaurant
from delivery_sim.repositories.restaurant_repository import RestaurantRepository
from delivery_sim.simulation.rng_manager import StructuralRNGManager
from delivery_sim.utils.logging_system import get_logger
from delivery_sim.utils.location_utils import format_location


class Infrastructure:
    """
    Complete physical delivery environment as a first-class entity.
    
    Infrastructure encapsulates all structural components of the delivery system:
    restaurant locations, area characteristics, and spatial arrangements.
    Once created, an Infrastructure instance can be reused across multiple
    experimental configurations that vary only operational parameters.
    """
    
    def __init__(self, structural_config, master_seed):
        """
        Initialize Infrastructure with deterministic generation.
        
        Args:
            structural_config: StructuralConfig defining area size, restaurant count, etc.
            master_seed: Master seed for deterministic infrastructure generation
        """
        self.logger = get_logger("infrastructure.core")
        
        # Store configuration for reference and validation
        self.structural_config = structural_config
        self.master_seed = master_seed
        
        # Initialize structural RNG for deterministic generation
        self.structural_rng = StructuralRNGManager(master_seed)
        
        # Initialize restaurant repository
        self.restaurant_repository = RestaurantRepository()
        
        # Analysis results cache (populated by InfrastructureAnalyzer)
        self.analysis_results = None
        
        # Generate infrastructure components
        self._generate_infrastructure()
        
        self.logger.info(f"Infrastructure created: {self}")
    
    def _generate_infrastructure(self):
        """Generate all infrastructure components deterministically."""
        self.logger.debug("Generating infrastructure components...")
        
        # Generate restaurants using structural RNG
        restaurants = self._generate_restaurants()
        
        # Add restaurants to repository
        for restaurant in restaurants:
            self.restaurant_repository.add(restaurant)
        
        self.logger.info(f"Generated {len(restaurants)} restaurants in "
                        f"{self.structural_config.delivery_area_size}x{self.structural_config.delivery_area_size}km area")
    
    def _generate_restaurants(self):
        """
        Generate restaurant locations using uniform random distribution.
        
        This method implements the methodological decision to use uniform random
        distribution for restaurant placement to create a spatially neutral
        simulation environment as outlined in the research methodology.
        
        Returns:
            list: Generated Restaurant objects with deterministic locations
        """
        restaurants = []
        area_size = self.structural_config.delivery_area_size
        num_restaurants = self.structural_config.num_restaurants
        
        self.logger.debug(f"Generating {num_restaurants} restaurants in {area_size}x{area_size}km area")
        
        for i in range(num_restaurants):
            # Generate deterministic location using structural RNG
            location = self.structural_rng.rng.uniform(0, area_size, size=2).tolist()
            restaurant_id = f"R{i+1}"
            
            restaurant = Restaurant(restaurant_id=restaurant_id, location=location)
            restaurants.append(restaurant)
            
            # Log restaurant creation (debug level for infrastructure)
            self.logger.debug(f"Generated restaurant {restaurant_id} at {format_location(location)}")
        
        return restaurants
    
    # ===== Public Interface for Analysis and Reuse =====
    
    def get_restaurant_repository(self):
        """
        Get the restaurant repository for this infrastructure.
        
        Returns:
            RestaurantRepository: Repository containing all restaurants
        """
        return self.restaurant_repository
    
    def get_structural_rng(self):
        """
        Get the structural RNG manager for this infrastructure.
        
        This enables analysis methods that need deterministic random sampling
        (e.g., Monte Carlo distance calculations) to use the same random stream.
        
        Returns:
            StructuralRNGManager: The RNG manager for this infrastructure
        """
        return self.structural_rng
    
    def get_basic_characteristics(self):
        """
        Get basic infrastructure characteristics without analysis.
        
        Returns basic derived metrics that don't require expensive computation.
        
        Returns:
            dict: Basic characteristics (area, count, density, spacing)
        """
        area_size = self.structural_config.delivery_area_size
        restaurant_count = len(self.restaurant_repository.find_all())
        restaurant_density = restaurant_count / (area_size ** 2)
        average_spacing = (area_size ** 2 / restaurant_count) ** 0.5
        
        return {
            'area_size': area_size,
            'restaurant_count': restaurant_count,
            'restaurant_density': restaurant_density,
            'average_restaurant_spacing': average_spacing,
            'driver_speed': self.structural_config.driver_speed
        }
    
    def set_analysis_results(self, analysis_results):
        """
        Store analysis results from InfrastructureAnalyzer.
        
        This enables Infrastructure to cache expensive analysis results
        for reuse across multiple experimental configurations.
        
        Args:
            analysis_results: Complete analysis results from InfrastructureAnalyzer
        """
        self.analysis_results = analysis_results
        self.logger.debug("Analysis results cached in infrastructure instance")
    
    def get_analysis_results(self):
        """
        Get cached analysis results if available.
        
        Returns:
            dict: Analysis results or None if not yet analyzed
        """
        return self.analysis_results
    
    def has_analysis_results(self):
        """
        Check if this infrastructure has been analyzed.
        
        Returns:
            bool: True if analysis results are available
        """
        return self.analysis_results is not None
    
    # ===== Infrastructure Identity and Validation =====
    
    def get_infrastructure_signature(self):
        """
        Generate unique signature for this infrastructure configuration.
        
        This enables experimental designs to verify they're using consistent
        infrastructure across multiple configurations.
        
        Returns:
            dict: Infrastructure signature for validation
        """
        return {
            'area_size': self.structural_config.delivery_area_size,
            'num_restaurants': self.structural_config.num_restaurants,
            'driver_speed': self.structural_config.driver_speed,
            'master_seed': self.master_seed
        }
    
    def is_compatible_with(self, other_infrastructure):
        """
        Check if this infrastructure is identical to another.
        
        Useful for experimental validation to ensure configurations
        are truly using the same infrastructure.
        
        Args:
            other_infrastructure: Another Infrastructure instance
            
        Returns:
            bool: True if infrastructures are identical
        """
        return (self.get_infrastructure_signature() == 
                other_infrastructure.get_infrastructure_signature())
    
    def __str__(self):
        """String representation for logging and debugging."""
        basic_chars = self.get_basic_characteristics()
        analysis_status = "analyzed" if self.has_analysis_results() else "not analyzed"
        
        return (f"Infrastructure("
                f"area={basic_chars['area_size']}x{basic_chars['area_size']}km, "
                f"restaurants={basic_chars['restaurant_count']}, "
                f"density={basic_chars['restaurant_density']:.3f}/km², "
                f"seed={self.master_seed}, "
                f"{analysis_status})")
    
    def __repr__(self):
        """Technical representation for debugging."""
        return f"Infrastructure(structural_config={self.structural_config}, master_seed={self.master_seed})"


def create_infrastructure(structural_config, master_seed):
    """
    Factory function for creating Infrastructure instances.
    
    Convenience function that handles Infrastructure creation with
    clear parameter documentation.
    
    Args:
        structural_config: StructuralConfig defining physical environment
        master_seed: Master seed for deterministic generation
        
    Returns:
        Infrastructure: Complete infrastructure instance ready for use/analysis
    """
    return Infrastructure(structural_config, master_seed)