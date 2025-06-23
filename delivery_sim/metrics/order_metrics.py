# delivery_sim/metrics/order_metrics.py
"""
Order-level metrics calculation.

This module provides functions to calculate timing metrics for individual orders
based on their lifecycle events in the food delivery simulation.

Key metrics:
- Order waiting time: Time from arrival until assignment
- Order travel time: Time from assignment until delivery (pure logistics)
- Order fulfillment time: Total time from arrival until delivery
"""

def calculate_order_waiting_time(order):
    """
    Calculate time from order arrival to driver assignment.
    
    This represents the customer's waiting experience before their order
    enters the delivery pipeline.
    
    Args:
        order: Order entity with arrival_time and assignment_time attributes
        
    Returns:
        float: Waiting time in simulation time units, or None if not assigned
    """
    if not order.assignment_time:
        return None
    return order.assignment_time - order.arrival_time


def calculate_order_travel_time(order):
    """
    Calculate time from driver assignment to delivery completion.
    
    This represents pure logistics time: driver travel to restaurant
    plus driver travel to customer (assuming instant pickup/handoff).
    
    Args:
        order: Order entity with assignment_time and delivery_time attributes
        
    Returns:
        float: Travel time in simulation time units, or None if not delivered
    """
    if not (order.assignment_time and order.delivery_time):
        return None
    return order.delivery_time - order.assignment_time


def calculate_order_fulfillment_time(order):
    """
    Calculate total time from order placement to delivery completion.
    
    This represents the complete customer experience from placing
    the order until receiving their food.
    
    Args:
        order: Order entity with arrival_time and delivery_time attributes
        
    Returns:
        float: Fulfillment time in simulation time units, or None if not delivered
    """
    if not order.delivery_time:
        return None
    return order.delivery_time - order.arrival_time


def calculate_all_order_metrics(order):
    """
    Calculate all basic timing metrics for a single order.
    
    Convenience function that calculates all three timing metrics
    for an order in a single call.
    
    Args:
        order: Order entity with timing attributes
        
    Returns:
        dict: Dictionary with metric names as keys and values as calculated times,
              or None for metrics that cannot be calculated (incomplete orders)
    """
    return {
        'waiting_time': calculate_order_waiting_time(order),
        'travel_time': calculate_order_travel_time(order),
        'fulfillment_time': calculate_order_fulfillment_time(order)
    }