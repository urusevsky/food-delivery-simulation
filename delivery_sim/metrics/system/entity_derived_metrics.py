# delivery_sim/metrics/system/entity_derived_metrics.py
"""
Entity-derived system metrics calculation.

This module provides functions to calculate system-level metrics that can be
derived entirely from entity repository data post-simulation, without needing
time-series collection during simulation.

Enhanced with pairing effectiveness metric for pairing parameter optimization.
"""


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


def calculate_pairing_effectiveness(repositories, warmup_period):
    """
    Calculate pairing effectiveness as proportion of arrived orders that were paired.
    
    This metric measures how effectively the pairing system is working by showing
    what percentage of orders that arrived during the analysis period got paired
    with another order, regardless of whether they were ultimately completed.
    
    Args:
        repositories: Dict containing original repositories
        warmup_period: Warmup period used for filtering
        
    Returns:
        dict: Contains total_arrived, total_paired, and pairing_effectiveness
    """
    from delivery_sim.analysis_pipeline.data_preparation import count_orders_arrived_during_analysis
    
    # Total orders that arrived during analysis period (regardless of completion or pairing)
    total_arrived = count_orders_arrived_during_analysis(repositories['order'], warmup_period)
    
    # Count orders that arrived during analysis period AND were paired
    all_orders = repositories['order'].find_all()
    paired_orders = [
        order for order in all_orders 
        if (order.arrival_time >= warmup_period and 
            order.pair is not None)
    ]
    total_paired = len(paired_orders)
    
    # Calculate pairing effectiveness
    pairing_effectiveness = total_paired / total_arrived if total_arrived > 0 else 0.0
    
    return {
        'total_arrived': total_arrived,
        'total_paired': total_paired,
        'pairing_effectiveness': pairing_effectiveness
    }





def calculate_all_entity_derived_system_metrics(repositories, filtered_entities, warmup_period):
    """
    Calculate all entity-derived system metrics for a replication.
    
    Enhanced with pairing effectiveness metric for pairing parameter optimization.
    
    Args:
        repositories: Dict containing original repositories
        filtered_entities: Dict containing pre-filtered entity lists from data_preparation
        warmup_period: Warmup period used for filtering
        
    Returns:
        dict: Dictionary with metric names as keys and calculated values
    """
    # Existing completion metrics
    completion_metrics = calculate_system_completion_rate(repositories, filtered_entities, warmup_period)
    
    # New pairing effectiveness metrics
    pairing_metrics = calculate_pairing_effectiveness(repositories, warmup_period)
    
    return {
        # Completion metrics (existing)
        'system_completion_rate': completion_metrics['completion_rate'],
        'total_orders_arrived': completion_metrics['total_arrived'],
        'total_orders_delivered': completion_metrics['total_delivered'],
        
        # Pairing effectiveness metrics (new)
        'pairing_effectiveness': pairing_metrics['pairing_effectiveness'],
        'total_orders_paired': pairing_metrics['total_paired']
    }