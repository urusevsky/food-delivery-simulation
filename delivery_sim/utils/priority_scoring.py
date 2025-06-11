# delivery_sim/utils/priority_scoring.py
"""
Priority Scoring System for Food Delivery Assignment Decisions

This module implements a multi-criteria scoring system that evaluates
driver-entity assignment opportunities based on distance efficiency,
throughput optimization, and fairness considerations.

Configuration is handled by ScoringConfig in configuration.py.
"""

import numpy as np
from delivery_sim.utils.location_utils import calculate_distance
from delivery_sim.utils.entity_type_utils import EntityType
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
        return typical_distance


class PriorityScorer:
    """
    Main scoring system that evaluates driver-entity assignment opportunities.
    """
    
    def __init__(self, scoring_config, typical_distance, env):
        """
        Initialize scorer with configuration and infrastructure characteristics.
        
        Args:
            scoring_config: ScoringConfig instance from configuration.py
            typical_distance: Calculated typical distance for this infrastructure
            env: SimPy environment for logging
        """
        self.config = scoring_config
        self.typical_distance = typical_distance
        self.env = env
        self.logger = get_logger()
    
    def calculate_priority_score(self, driver, entity):
        """
        Calculate priority score for a driver-entity assignment.
        
        Args:
            driver: Driver being evaluated
            entity: Order or Pair being evaluated
            
        Returns:
            tuple: (priority_score_0_to_100, components_dictionary)
        """
        # Calculate individual component scores (all in [0,1] range)
        distance_score = self._calculate_distance_score(driver, entity)
        throughput_score = self._calculate_throughput_score(entity)
        fairness_score = self._calculate_fairness_score(entity)
        
        # Weighted combination
        combined_score = (
            self.config.weight_distance * distance_score +
            self.config.weight_throughput * throughput_score +
            self.config.weight_fairness * fairness_score
        )
        
        # Scale to 0-100 range for interpretability
        priority_score = combined_score * 100
        
        # Log detailed calculation
        self.logger.debug(
            f"[t={self.env.now:.2f}] Priority score calculation: "
            f"distance={distance_score:.3f}, throughput={throughput_score:.3f}, "
            f"fairness={fairness_score:.3f}, combined={priority_score:.2f}"
        )
        
        # Return score and components for logging/analysis
        components = {
            "distance_score": distance_score,
            "throughput_score": throughput_score,
            "fairness_score": fairness_score,
            "combined_score_0_1": combined_score,
            "total_distance": self._calculate_total_distance(driver, entity),
            "num_orders": self._get_order_count(entity),
            "wait_time_minutes": self._calculate_wait_time(entity)
        }
        
        return priority_score, components
    
    def _calculate_distance_score(self, driver, entity):
        """
        Calculate distance efficiency score (0-1, higher = better).
        
        Uses two-step normalization for relational measures:
        1. Contextualization: actual_distance / typical_distance  
        2. Performance assessment: apply universal standard
        """
        actual_distance = self._calculate_total_distance(driver, entity)
        
        # Step 1: Contextualization (acknowledge geographical reality)
        distance_ratio = actual_distance / self.typical_distance
        
        # Step 2: Performance assessment (apply universal standard)
        # max(0, ...) ensures score doesn't go below 0
        distance_score = max(0, 1 - distance_ratio / self.config.max_distance_ratio_multiplier)
        
        return distance_score
    
    def _calculate_throughput_score(self, entity):
        """
        Calculate throughput value score (0-1, higher = better).
        
        Direct normalization since order count is absolute measure.
        """
        num_orders = self._get_order_count(entity)
        throughput_score = num_orders / self.config.max_orders_per_trip
        
        return throughput_score
    
    def _calculate_fairness_score(self, entity):
        """
        Calculate fairness (wait time) score (0-1, higher urgency = higher score).
        
        Direct normalization since time is absolute measure.
        """
        wait_time = self._calculate_wait_time(entity)
        
        # min(1.0, ...) ensures score doesn't exceed 1
        fairness_score = min(1.0, wait_time / self.config.max_acceptable_wait)
        
        return fairness_score
    
    def _calculate_total_distance(self, driver, entity):
        """Calculate total delivery distance for driver-entity assignment."""
        if entity.entity_type == EntityType.ORDER:
            # Driver -> Restaurant -> Customer
            driver_to_restaurant = calculate_distance(
                driver.current_location, entity.restaurant_location
            )
            restaurant_to_customer = calculate_distance(
                entity.restaurant_location, entity.customer_location
            )
            return driver_to_restaurant + restaurant_to_customer
        
        elif entity.entity_type == EntityType.PAIR:
            # Use pre-calculated optimal route for pairs
            driver_to_start = calculate_distance(
                driver.current_location, entity.optimal_sequence[0]
            )
            return driver_to_start + entity.optimal_cost
        
        else:
            raise ValueError(f"Unknown entity type: {entity.entity_type}")
    
    def _get_order_count(self, entity):
        """Get number of orders in this delivery entity."""
        return 2 if entity.entity_type == EntityType.PAIR else 1
    
    def _calculate_wait_time(self, entity):
        """Calculate wait time in minutes for this entity."""
        if entity.entity_type == EntityType.PAIR:
            # Use earliest arrival time for pairs
            arrival_time = min(entity.order1.arrival_time, entity.order2.arrival_time)
        else:
            arrival_time = entity.arrival_time
        
        wait_time_minutes = self.env.now - arrival_time
        return wait_time_minutes


def create_priority_scorer(restaurant_repository, area_size, structural_rng, env, scoring_config):
    """
    Factory function to create a PriorityScorer with calculated typical_distance.
    
    Args:
        restaurant_repository: Repository containing restaurant locations
        area_size: Size of the square delivery area
        structural_rng: StructuralRNGManager for reproducible random sampling
        env: SimPy environment
        scoring_config: ScoringConfig instance from configuration.py
        
    Returns:
        PriorityScorer: Configured scoring system
    """
    # Calculate typical distance for this infrastructure
    typical_distance = TypicalDistanceCalculator.calculate_typical_distance(
        restaurant_repository, area_size, structural_rng, scoring_config.typical_distance_samples
    )
    
    # Create and return scorer directly
    return PriorityScorer(scoring_config, typical_distance, env)