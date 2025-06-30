# delivery_sim/simulation/data_preparation.py
"""
Data preparation and filtering for metrics analysis.

This module provides functions to filter raw simulation data and prepare it
for metrics calculation, including warmup period filtering and entity validation.
"""


def filter_entities_for_analysis(repositories, warmup_period):
    """
    Filter entities from repositories to exclude warmup period data.
    
    This function applies warmup filtering to different entity types based on
    their relevant timing attributes, preparing clean datasets for metrics analysis.
    
    Args:
        repositories: Dict containing entity repositories
        warmup_period: Duration (in simulation time) to exclude from analysis
        
    Returns:
        dict: Filtered entities by type, ready for metrics calculation
    """
    filtered_entities = {}
    
    # Filter orders: arrived after warmup AND completed (full lifecycle)
    if 'order' in repositories:
        filtered_entities['order'] = _filter_orders_by_warmup(
            repositories['order'], warmup_period
        )
    
    # Filter delivery units: constituent orders arrived after warmup AND delivery completed
    if 'delivery_unit' in repositories:
        filtered_entities['delivery_unit'] = _filter_delivery_units_by_warmup(
            repositories['delivery_unit'], warmup_period
        )
    
    return filtered_entities


def count_orders_arrived_during_analysis(order_repository, warmup_period):
    """
    Count orders that arrived during analysis period (regardless of completion status).
    
    This is used for calculating completion rates in system metrics.
    
    Args:
        order_repository: Repository containing all orders
        warmup_period: Duration to exclude from start
        
    Returns:
        int: Number of orders that arrived after warmup period
    """
    all_orders = order_repository.find_all()
    return len([
        order for order in all_orders 
        if order.arrival_time >= warmup_period
    ])


def _filter_orders_by_warmup(order_repository, warmup_period):
    """
    Filter orders that arrived after warmup period AND were delivered (complete lifecycle).
    """
    all_orders = order_repository.find_all()
    return [
        order for order in all_orders 
        if (order.arrival_time >= warmup_period and 
            order.delivery_time is not None)
    ]


def _filter_delivery_units_by_warmup(delivery_unit_repository, warmup_period):
    """
    Filter delivery units where ALL constituent orders arrived after warmup period
    AND the delivery unit was completed (complete lifecycle).
    
    This ensures delivery units represent complete lifecycles during the 
    representative period, not hybrid startup/steady-state behavior.
    """
    from delivery_sim.utils.entity_type_utils import EntityType
    
    all_units = delivery_unit_repository.find_all()
    valid_units = []
    
    for unit in all_units:
        # Must be completed (complete lifecycle requirement)
        if not unit.completion_time:
            continue
            
        entity = unit.delivery_entity
        
        if entity.entity_type == EntityType.ORDER:
            # Single order: order must have arrived after warmup
            if entity.arrival_time >= warmup_period:
                valid_units.append(unit)
                
        elif entity.entity_type == EntityType.PAIR:
            # Paired orders: BOTH orders must have arrived after warmup
            if (entity.order1.arrival_time >= warmup_period and 
                entity.order2.arrival_time >= warmup_period):
                valid_units.append(unit)
    
    return valid_units


def get_analysis_time_window(simulation_duration, warmup_period):
    """
    Calculate effective analysis time window after warmup exclusion.
    
    Args:
        simulation_duration: Total simulation duration
        warmup_period: Duration to exclude from start
        
    Returns:
        tuple: (analysis_start_time, analysis_end_time)
    """
    analysis_start_time = warmup_period
    analysis_end_time = simulation_duration
    
    if analysis_start_time >= analysis_end_time:
        raise ValueError(f"Warmup period ({warmup_period}) must be less than simulation duration ({simulation_duration})")
    
    return analysis_start_time, analysis_end_time