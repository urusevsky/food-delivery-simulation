# delivery_sim/simulation/infrastructure_analysis.py
"""
Infrastructure Analysis Module for Food Delivery Simulation

This module performs expensive infrastructure analysis (like Monte Carlo sampling for typical distances)
once per configuration. Results are cached and reused across multiple replications for efficiency.

The analysis captures the geographical and structural reality of the delivery environment,
providing infrastructure-derived normalizers for scoring systems.
"""

import numpy as np
from delivery_sim.utils.location_utils import calculate_distance
from delivery_sim.utils.logging_system import get_logger


class TypicalDistanceCalculator:
    """
    Calculates infrastructure-derived normalizers through Monte Carlo sampling.
    """
    
    @staticmethod
    def calculate_typical_distance(restaurant_repository, area_size, structural_rng, sample_size=1000):
        """
        Derives the characteristic single-order delivery distance
        for a given infrastructure through Monte Carlo sampling.
        
        Args:
            restaurant_repository: Repository containing restaurant locations
            area_size: Size of the square delivery area
            structural_rng: StructuralRNGManager for reproducible sampling
            sample_size: Number of samples for Monte Carlo estimation
            
        Returns:
            float: Typical distance for this geographic configuration
        """
        samples = []
        restaurants = restaurant_repository.find_all()
        
        if not restaurants:
            raise ValueError("No restaurants found in repository for distance calculation")
        
        logger = get_logger()
        logger.debug(f"Calculating typical distance with {sample_size} samples")
        logger.debug(f"Area: {area_size}x{area_size}km, Restaurants: {len(restaurants)}")
        
        for _ in range(sample_size):
            # Sample driver location (uniform in delivery area)
            driver_loc = structural_rng.rng.uniform(0, area_size, size=2).tolist()
            
            # Sample restaurant (uniform selection)
            restaurant = structural_rng.rng.choice(restaurants)
            
            # Sample customer location (uniform in delivery area)
            customer_loc = structural_rng.rng.uniform(0, area_size, size=2).tolist()
            
            # Calculate full delivery distance
            distance = (
                calculate_distance(driver_loc, restaurant.location) +
                calculate_distance(restaurant.location, customer_loc)
            )
            samples.append(distance)
        
        # Use median for robustness to outliers
        typical_distance = np.median(samples)
        
        logger.debug(f"Typical distance calculated: {typical_distance:.3f}km")
        logger.debug(f"Distance distribution: min={np.min(samples):.3f}, "
                    f"mean={np.mean(samples):.3f}, max={np.max(samples):.3f}")
        
        return typical_distance


def analyze_infrastructure(restaurant_repository, structural_config, structural_rng, scoring_config=None):
    """
    Perform comprehensive infrastructure analysis for a delivery system configuration.
    
    This function orchestrates expensive analysis operations (like Monte Carlo sampling)
    that characterize the infrastructure. Results are cached and reused across replications.
    
    Args:
        restaurant_repository: Repository containing restaurant locations
        structural_config: StructuralConfig with area_size, num_restaurants, etc.
        structural_rng: StructuralRNGManager for reproducible random sampling
        scoring_config: Optional ScoringConfig for advanced analysis parameters
        
    Returns:
        dict: Infrastructure characteristics containing:
            - area_size: Delivery area size
            - restaurant_count: Number of restaurants
            - typical_distance: Characteristic delivery distance for this geography
            - restaurant_density: Restaurants per square km
            - average_restaurant_spacing: Rough spacing between restaurants
    """
    logger = get_logger()
    logger.info("Starting infrastructure analysis...")
    
    # Basic structural characteristics
    area_size = structural_config.delivery_area_size
    restaurants = restaurant_repository.find_all()
    restaurant_count = len(restaurants)
    
    if restaurant_count == 0:
        raise ValueError("Cannot analyze infrastructure with no restaurants")
    
    # Calculate derived metrics
    restaurant_density = restaurant_count / (area_size ** 2)
    average_restaurant_spacing = (area_size ** 2 / restaurant_count) ** 0.5
    
    logger.debug(f"Basic infrastructure: {area_size}x{area_size}km area, {restaurant_count} restaurants")
    logger.debug(f"Restaurant density: {restaurant_density:.3f} restaurants/km²")
    logger.debug(f"Average restaurant spacing: {average_restaurant_spacing:.3f}km")
    
    # Calculate typical distance through Monte Carlo sampling
    sample_size = getattr(scoring_config, 'typical_distance_samples', 1000) if scoring_config else 1000
    typical_distance = TypicalDistanceCalculator.calculate_typical_distance(
        restaurant_repository=restaurant_repository,
        area_size=area_size,
        structural_rng=structural_rng,
        sample_size=sample_size
    )
    
    # Assemble infrastructure characteristics
    infrastructure_characteristics = {
        'area_size': area_size,
        'restaurant_count': restaurant_count,
        'typical_distance': typical_distance,
        'restaurant_density': restaurant_density,
        'average_restaurant_spacing': average_restaurant_spacing,
        'analysis_sample_size': sample_size
    }
    
    logger.info(f"Infrastructure analysis complete: typical_distance={typical_distance:.3f}km, "
               f"density={restaurant_density:.3f}/km², spacing={average_restaurant_spacing:.3f}km")
    
    return infrastructure_characteristics