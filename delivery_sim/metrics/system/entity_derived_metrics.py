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


def calculate_system_completion_rate(repositories, filtered_entities, warmup_period):
    """
    Calculate system completion rate as proportion of arrived orders that were delivered.
    
    This metric provides the primary measure of system performance by showing what 
    percentage of orders that arrived during the analysis period were successfully completed.
    
    Args:
        repositories: Dict containing original repositories (for counting all arrivals)
        filtered_entities: Dict containing filtered entity lists (for counting completions)
        warmup_period: Warmup period used for filtering
        
    Returns:
        dict: Contains total_arrived, total_delivered, and completion_rate
    """
    from delivery_sim.analysis_pipeline.data_preparation import count_orders_arrived_during_analysis
    
    # Total orders that arrived during analysis period (regardless of completion)
    total_arrived = count_orders_arrived_during_analysis(repositories['order'], warmup_period)
    
    # Total orders that arrived during analysis period AND were completed (filtered orders)
    filtered_orders = filtered_entities.get('order', [])
    total_delivered = len(filtered_orders)
    
    # Calculate completion rate
    completion_rate = total_delivered / total_arrived if total_arrived > 0 else 0.0
    
    return {
        'total_arrived': total_arrived,
        'total_delivered': total_delivered, 
        'completion_rate': completion_rate
    }


def calculate_all_entity_derived_system_metrics(repositories, filtered_entities, warmup_period):
    """
    Calculate all entity-derived system metrics for a replication.
    
    Convenience function that calculates all system metrics using
    pre-filtered entities from data preparation.
    
    Args:
        repositories: Dict containing original repositories
        filtered_entities: Dict containing pre-filtered entity lists from data_preparation
        warmup_period: Warmup period used for filtering
        
    Returns:
        dict: Dictionary with metric names as keys and calculated values
    """
    filtered_delivery_units = filtered_entities.get('delivery_unit', [])
    
    throughput = calculate_system_throughput(filtered_delivery_units)
    completion_metrics = calculate_system_completion_rate(repositories, filtered_entities, warmup_period)
    
    return {
        'system_throughput': throughput,
        'system_completion_rate': completion_metrics['completion_rate'],
        'total_orders_arrived': completion_metrics['total_arrived'],
        'total_orders_delivered': completion_metrics['total_delivered']
    }