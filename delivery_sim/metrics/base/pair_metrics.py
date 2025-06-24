# delivery_sim/metrics/pair_metrics.py
"""
Pair-level metrics calculation.

This module provides functions to calculate timing metrics for pairs of orders
based on their lifecycle events in the food delivery simulation.

Key metrics:
- Pair formation time: Time from first order arrival to pair formation
- Pair assignment time: Time from pair formation to driver assignment
- Pair travel time: Time from assignment to completion of both orders
- Pair fulfillment time: Total time from first order arrival to pair completion
- Average order fulfillment time: Average fulfillment time of constituent orders
- Pair sequential delay: Time between first and second delivery completion
"""

def calculate_pair_formation_time(pair):
    """
    Calculate time from first order arrival to pair formation.
    
    This represents how long it took to identify and form a suitable
    pair from the constituent orders.
    
    Args:
        pair: Pair entity with order1, order2 references and creation_time
        
    Returns:
        float: Formation time in simulation time units
    """
    # Find the earlier arrival time between the two orders
    first_arrival_time = min(pair.order1.arrival_time, pair.order2.arrival_time)
    return pair.creation_time - first_arrival_time


def calculate_pair_assignment_time(pair):
    """
    Calculate time from pair formation to driver assignment.
    
    This represents how long the formed pair waited in the system
    before being assigned to a driver.
    
    Args:
        pair: Pair entity with creation_time and assignment_time attributes
        
    Returns:
        float: Assignment time in simulation time units, or None if not assigned
    """
    if not pair.assignment_time:
        return None
    return pair.assignment_time - pair.creation_time


def calculate_pair_travel_time(pair):
    """
    Calculate time from driver assignment to completion of both orders.
    
    This represents pure logistics time for delivering both orders
    in the pair (assuming instant pickup/handoff at restaurants).
    
    Args:
        pair: Pair entity with assignment_time and completion_time attributes
        
    Returns:
        float: Travel time in simulation time units, or None if not completed
    """
    if not (pair.assignment_time and pair.completion_time):
        return None
    return pair.completion_time - pair.assignment_time


def calculate_pair_fulfillment_time(pair):
    """
    Calculate total time from first order arrival to pair completion.
    
    This represents the complete experience from when the first
    order was placed until both orders in the pair were delivered.
    
    Args:
        pair: Pair entity with order1, order2 references and completion_time
        
    Returns:
        float: Fulfillment time in simulation time units, or None if not completed
    """
    if not pair.completion_time:
        return None
    # Find the earlier arrival time between the two orders
    first_arrival_time = min(pair.order1.arrival_time, pair.order2.arrival_time)
    return pair.completion_time - first_arrival_time


def calculate_average_order_fulfillment_time_for_pair(pair):
    """
    Calculate average fulfillment time of the two orders in the pair.
    
    This represents the average individual customer experience
    within the pair delivery.
    
    Args:
        pair: Pair entity with order1, order2 references having delivery_time
        
    Returns:
        float: Average fulfillment time in simulation time units, or None if either order not delivered
    """
    if not (pair.order1.delivery_time and pair.order2.delivery_time):
        return None
    
    order1_fulfillment = pair.order1.delivery_time - pair.order1.arrival_time
    order2_fulfillment = pair.order2.delivery_time - pair.order2.arrival_time
    
    return (order1_fulfillment + order2_fulfillment) / 2


def calculate_pair_sequential_delay(pair):
    """
    Calculate time between first and second delivery completion in the pair.
    
    This represents the coordination cost where one customer must wait
    for the other customer's delivery to complete first.
    
    Args:
        pair: Pair entity with order1, order2 references having delivery_time
        
    Returns:
        float: Sequential delay in simulation time units, or None if either order not delivered
    """
    if not (pair.order1.delivery_time and pair.order2.delivery_time):
        return None
    
    # Find which order was delivered first and which was delivered last
    first_delivery_time = min(pair.order1.delivery_time, pair.order2.delivery_time)
    last_delivery_time = max(pair.order1.delivery_time, pair.order2.delivery_time)
    
    return last_delivery_time - first_delivery_time


def calculate_all_pair_metrics(pair):
    """
    Calculate all basic timing metrics for a single pair.
    
    Convenience function that calculates all six timing metrics
    for a pair in a single call.
    
    Args:
        pair: Pair entity with timing attributes and order references
        
    Returns:
        dict: Dictionary with metric names as keys and values as calculated times,
              or None for metrics that cannot be calculated (incomplete pairs)
    """
    return {
        'formation_time': calculate_pair_formation_time(pair),
        'assignment_time': calculate_pair_assignment_time(pair),
        'travel_time': calculate_pair_travel_time(pair),
        'fulfillment_time': calculate_pair_fulfillment_time(pair),
        'average_order_fulfillment_time': calculate_average_order_fulfillment_time_for_pair(pair),
        'sequential_delay': calculate_pair_sequential_delay(pair)
    }