# delivery_sim/metrics/system/entity_derived_metrics.py
"""
Entity-derived system metrics calculation.

This module provides functions to calculate system-level metrics that can be
derived entirely from entity repository data post-simulation, without needing
time-series collection during simulation.

Key metrics:
- System throughput: Orders delivered per unit time over analysis window
"""


def calculate_system_throughput(filtered_delivery_units):
    """
    Calculate system throughput as total orders delivered during analysis period.
    
    This metric measures how many orders the system successfully delivered from
    demand that arrived during the representative analysis period.
    
    Args:
        filtered_delivery_units: List of delivery units already filtered by data preparation
        
    Returns:
        int: Total orders delivered during analysis period
    """
    # Sum total orders delivered (each unit contains 1 or 2 orders)
    total_orders = sum(
        unit.assignment_scores.get("num_orders", 0) 
        for unit in filtered_delivery_units
        if unit.assignment_scores
    )
    
    return total_orders


def calculate_all_entity_derived_system_metrics(filtered_entities):
    """
    Calculate all entity-derived system metrics for a replication.
    
    Convenience function that calculates all system metrics using
    pre-filtered entities from data preparation.
    
    Args:
        filtered_entities: Dict containing pre-filtered entity lists from data_preparation
        
    Returns:
        dict: Dictionary with metric names as keys and calculated values
    """
    filtered_delivery_units = filtered_entities.get('delivery_unit', [])
    
    return {
        'system_throughput': calculate_system_throughput(filtered_delivery_units)
    }