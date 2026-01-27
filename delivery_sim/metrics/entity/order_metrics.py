# delivery_sim/metrics/order_metrics.py
"""
Order-level metrics calculation.

This module provides functions to calculate timing metrics for individual orders
based on their lifecycle events in the food delivery simulation.

Key metrics:
- Order assignment time: Time from arrival until assignment
- Order pickup travel time: Time from assignment until pickup (driver → restaurant)
- Order delivery travel time: Time from pickup until delivery (restaurant → customer)
- Order travel time: Time from assignment until delivery (pure logistics)
- Order fulfillment time: Total time from arrival until delivery
"""

def calculate_order_assignment_time(order):
    """
    Calculate time from order arrival to driver assignment.
    
    This represents how long it took for an order to get assigned.
    
    Args:
        order: Order entity with arrival_time and assignment_time attributes
        
    Returns:
        float: Waiting time in simulation time units, or None if not assigned
    """
    if not order.assignment_time:
        return None
    return order.assignment_time - order.arrival_time


def calculate_order_pickup_travel_time(order):
    """
    Calculate time from driver assignment to food pickup at restaurant.
    
    This represents the first leg of logistics: driver traveling to restaurant
    to pick up the prepared food.
    
    Args:
        order: Order entity with assignment_time and pickup_time attributes
        
    Returns:
        float: Pickup travel time in simulation time units, or None if not picked up
    """
    if not (order.assignment_time and order.pickup_time):
        return None
    return order.pickup_time - order.assignment_time


def calculate_order_delivery_travel_time(order):
    """
    Calculate time from food pickup to delivery completion at customer.
    
    This represents the second leg of logistics: driver traveling from restaurant
    to customer with the food.
    
    Args:
        order: Order entity with pickup_time and delivery_time attributes
        
    Returns:
        float: Delivery travel time in simulation time units, or None if not delivered
    """
    if not (order.pickup_time and order.delivery_time):
        return None
    return order.delivery_time - order.pickup_time


def calculate_order_travel_time(order):
    """
    Calculate time from order assignment to delivery completion.
    
    This represents pure logistics time from order assignment to order completion.
    Mathematically equals: pickup_travel_time + delivery_travel_time
    
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
    Mathematically equals: assignment_time + travel_time
    
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
    
    Convenience function that calculates all five timing metrics
    for an order in a single call.
    
    Args:
        order: Order entity with timing attributes
        
    Returns:
        dict: Dictionary with metric names as keys and values as calculated times,
              or None for metrics that cannot be calculated (incomplete orders)
    """
    return {
        'assignment_time': calculate_order_assignment_time(order),
        'pickup_travel_time': calculate_order_pickup_travel_time(order),
        'delivery_travel_time': calculate_order_delivery_travel_time(order),
        'travel_time': calculate_order_travel_time(order),
        'fulfillment_time': calculate_order_fulfillment_time(order)
    }