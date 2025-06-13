# delivery_sim/utils/priority_scoring.py
"""
Priority Scoring System for Food Delivery Assignment Decisions

This module implements a multi-criteria scoring system that evaluates
driver-entity assignment opportunities based on distance efficiency,
throughput optimization, and fairness considerations.

Configuration is handled by ScoringConfig in configuration.py.
Infrastructure analysis (like typical_distance calculation) is now handled 
by the infrastructure_analysis module for better separation of concerns.
"""

import numpy as np
from delivery_sim.utils.location_utils import calculate_distance
from delivery_sim.utils.entity_type_utils import EntityType
from delivery_sim.utils.logging_system import get_logger


class PriorityScorer:
    """
    Main scoring system that evaluates driver-entity assignment opportunities.
    
    This scorer is designed to be reusable across multiple replications by separating
    infrastructure-derived parameters (calculated once) from environment-specific
    components (updated per replication).
    """
    
    def __init__(self, scoring_config, typical_distance, env):
        """
        Initialize scorer with configuration and infrastructure characteristics.
        
        Args:
            scoring_config: ScoringConfig instance from configuration.py
            typical_distance: Pre-calculated typical distance for this infrastructure
            env: SimPy environment for this replication
        """
        self.config = scoring_config
        self.typical_distance = typical_distance
        self.env = env
        self.logger = get_logger()
        
        self.logger.debug(f"PriorityScorer initialized with typical_distance={typical_distance:.3f}km")
    
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
        2. Performance evaluation: good performance = low distance ratios
        """
        total_distance = self._calculate_total_distance(driver, entity)
        distance_ratio = total_distance / self.typical_distance
        
        # Transform to score: ratio of 1.0 = score of 1.0 (perfect)
        # Ratios above max_distance_ratio_multiplier = score of 0.0 (unacceptable)
        max_ratio = self.config.max_distance_ratio_multiplier
        
        if distance_ratio <= 1.0:
            # Better than typical: score above baseline
            score = 1.0
        elif distance_ratio >= max_ratio:
            # Unacceptably long: minimum score
            score = 0.0
        else:
            # Linear interpolation between 1.0 and 0.0
            score = 1.0 - (distance_ratio - 1.0) / (max_ratio - 1.0)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_throughput_score(self, entity):
        """
        Calculate throughput optimization score (0-1, higher = better).
        
        Absolute measure - same meaning regardless of geography.
        """
        num_orders = self._get_order_count(entity)
        max_orders = self.config.max_orders_per_trip
        
        # Linear scaling: more orders = higher score
        score = (num_orders - 1) / (max_orders - 1) if max_orders > 1 else 0.0
        
        return max(0.0, min(1.0, score))
    
    def _calculate_fairness_score(self, entity):
        """
        Calculate fairness score (0-1, higher = better for shorter waits).
        
        Absolute measure - same meaning regardless of geography.
        """
        wait_time = self._calculate_wait_time(entity)
        max_wait = self.config.max_acceptable_wait
        
        # Transform to score: no wait = 1.0, max_wait = 0.0
        if wait_time <= 0:
            score = 1.0
        elif wait_time >= max_wait:
            score = 0.0
        else:
            score = 1.0 - (wait_time / max_wait)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_total_distance(self, driver, entity):
        """Calculate total travel distance for this assignment."""
        if entity.entity_type == EntityType.PAIR:
            # Driver → Restaurant1 → Customer1 → Restaurant2 → Customer2
            distance = (
                calculate_distance(driver.location, entity.order1.restaurant.location) +
                calculate_distance(entity.order1.restaurant.location, entity.order1.customer_location) +
                calculate_distance(entity.order1.customer_location, entity.order2.restaurant.location) +
                calculate_distance(entity.order2.restaurant.location, entity.order2.customer_location)
            )
        else:
            # Driver → Restaurant → Customer
            distance = (
                calculate_distance(driver.location, entity.restaurant.location) +
                calculate_distance(entity.restaurant.location, entity.customer_location)
            )
        
        return distance
    
    def _get_order_count(self, entity):
        """Get number of orders in this entity."""
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


def create_priority_scorer(infrastructure_characteristics, scoring_config, env):
    """
    Factory function to create a PriorityScorer for a specific replication.
    
    This creates a variant component that uses pre-calculated infrastructure characteristics
    but is bound to a specific simulation environment.
    
    Args:
        infrastructure_characteristics: Dict containing 'typical_distance' and other metrics
        scoring_config: ScoringConfig instance from configuration.py
        env: SimPy environment for this replication
        
    Returns:
        PriorityScorer: Configured scoring system for this replication
    """
    typical_distance = infrastructure_characteristics['typical_distance']
    
    # Create scorer bound to this replication's environment
    scorer = PriorityScorer(
        scoring_config=scoring_config,
        typical_distance=typical_distance,
        env=env
    )
    
    return scorer