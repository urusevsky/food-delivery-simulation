# delivery_sim/metrics/data_preparation.py
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
    
    # Filter orders by arrival time (when they entered the system)
    if 'order' in repositories:
        filtered_entities['order'] = _filter_orders_by_warmup(
            repositories['order'], warmup_period
        )
    
    # Filter delivery units by completion time (when they finished)
    if 'delivery_unit' in repositories:
        filtered_entities['delivery_unit'] = _filter_delivery_units_by_warmup(
            repositories['delivery_unit'], warmup_period
        )
    
    # Filter drivers by login time (when they entered the system)
    if 'driver' in repositories:
        filtered_entities['driver'] = _filter_drivers_by_warmup(
            repositories['driver'], warmup_period
        )
    
    # Filter pairs by creation time (when they were formed)
    if 'pair' in repositories:
        filtered_entities['pair'] = _filter_pairs_by_warmup(
            repositories['pair'], warmup_period
        )
    
    return filtered_entities


def _filter_orders_by_warmup(order_repository, warmup_period):
    """Filter orders that arrived after warmup period."""
    all_orders = order_repository.find_all()
    return [
        order for order in all_orders 
        if order.arrival_time >= warmup_period
    ]


def _filter_delivery_units_by_warmup(delivery_unit_repository, warmup_period):
    """
    Filter delivery units where ALL constituent orders arrived after warmup period.
    
    This ensures delivery units represent complete lifecycles during the 
    representative period, not hybrid startup/steady-state behavior.
    """
    from delivery_sim.utils.entity_type_utils import EntityType
    
    all_units = delivery_unit_repository.find_all()
    valid_units = []
    
    for unit in all_units:
        # Must be completed
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


def _filter_drivers_by_warmup(driver_repository, warmup_period):
    """Filter drivers that logged in after warmup period."""
    all_drivers = driver_repository.find_all()
    return [
        driver for driver in all_drivers 
        if driver.login_time >= warmup_period
    ]


def _filter_pairs_by_warmup(pair_repository, warmup_period):
    """Filter pairs that were created after warmup period.""" 
    all_pairs = pair_repository.find_all()
    return [
        pair for pair in all_pairs 
        if pair.creation_time >= warmup_period
    ]


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