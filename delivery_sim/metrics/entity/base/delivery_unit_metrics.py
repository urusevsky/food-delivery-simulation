# delivery_sim/metrics/entity/base/delivery_unit_metrics.py
"""
Delivery unit-level metrics calculation.

This module provides functions to calculate metrics for individual delivery units
based on their properties and lifecycle events in the food delivery simulation.

Key metrics:
- Total distance: Total travel distance for the delivery
"""

def calculate_delivery_unit_total_distance(delivery_unit):
    """
    Calculate total travel distance for a delivery unit.
    
    This metric represents the complete travel cost for delivering
    the assigned order(s), including driver-to-restaurant and 
    restaurant-to-customer segments.
    
    Args:
        delivery_unit: DeliveryUnit entity with assignment_scores containing total_distance
        
    Returns:
        float: Total distance in km, or None if not available
    """
    if not delivery_unit.assignment_scores:
        return None
    return delivery_unit.assignment_scores.get("total_distance")


def calculate_all_delivery_unit_metrics(delivery_unit):
    """
    Calculate all basic metrics for a single delivery unit.
    
    Convenience function that calculates all delivery unit metrics
    in a single call.
    
    Args:
        delivery_unit: DeliveryUnit entity with assignment scores
        
    Returns:
        dict: Dictionary with metric names as keys and values as calculated metrics,
              or None for metrics that cannot be calculated
    """
    return {
        'total_distance': calculate_delivery_unit_total_distance(delivery_unit)
    }