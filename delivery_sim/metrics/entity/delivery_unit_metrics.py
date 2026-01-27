# delivery_sim/metrics/entity/delivery_unit_metrics.py
"""
Delivery unit-level metrics calculation.

This module provides functions to calculate performance metrics for delivery units
(the assignment contract between a driver and a delivery entity).

Key metrics:
- First contact time: Time from assignment to reaching the first restaurant
- Fulfillment time: Total time from assignment to completion
"""

from delivery_sim.utils.entity_type_utils import EntityType


def calculate_delivery_unit_first_contact_time(delivery_unit):
    """
    Calculate time from assignment to first restaurant contact.
    
    This measures how long it takes the driver to reach the FIRST
    restaurant in the delivery sequence, regardless of whether it's
    a single order or paired delivery.
    
    This metric is critical for studying the "nearest-restaurant effect":
    the hypothesis that increasing restaurant count reduces first contact
    time because drivers have a higher probability of being near a restaurant.
    
    Args:
        delivery_unit: DeliveryUnit entity with delivery_entity and assignment_time
        
    Returns:
        float: First contact time in simulation time units, or None if not completed
    """
    entity = delivery_unit.delivery_entity
    assignment_time = delivery_unit.assignment_time
    
    if entity.entity_type == EntityType.ORDER:
        # Single order: first contact is the only pickup
        if not entity.pickup_time:
            return None
        return entity.pickup_time - assignment_time
    
    else:  # EntityType.PAIR
        # Paired orders: first contact is whichever restaurant was reached first
        order1 = entity.order1
        order2 = entity.order2
        
        # Both orders must have pickup times
        if not (order1.pickup_time and order2.pickup_time):
            return None
        
        # Return time to first restaurant contact
        first_pickup_time = min(order1.pickup_time, order2.pickup_time)
        return first_pickup_time - assignment_time


def calculate_delivery_unit_fulfillment_time(delivery_unit):
    """
    Calculate total time from assignment to delivery completion.
    
    This represents the complete execution time of the delivery unit,
    from when the driver was assigned until all orders were delivered.
    
    Args:
        delivery_unit: DeliveryUnit entity with assignment_time and completion_time
        
    Returns:
        float: Fulfillment time in simulation time units, or None if not completed
    """
    if not delivery_unit.completion_time:
        return None
    return delivery_unit.completion_time - delivery_unit.assignment_time


def calculate_all_delivery_unit_metrics(delivery_unit):
    """
    Calculate all basic metrics for a single delivery unit.
    
    Convenience function that calculates all timing metrics
    for a delivery unit in a single call.
    
    Args:
        delivery_unit: DeliveryUnit entity with timing attributes
        
    Returns:
        dict: Dictionary with metric names as keys and values as calculated times,
              or None for metrics that cannot be calculated (incomplete delivery units)
    """
    return {
        'first_contact_time': calculate_delivery_unit_first_contact_time(delivery_unit),
        'fulfillment_time': calculate_delivery_unit_fulfillment_time(delivery_unit),
        # Note: total_distance is already stored as delivery_unit.assignment_scores['total_distance']
        # and can be accessed directly rather than calculated
    }