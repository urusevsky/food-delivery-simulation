class OrderState:
    """States that an order can be in throughout its lifecycle"""
    CREATED = "created"      # Just entered the system, waiting for matching/assignment
    PAIRED = "paired"        # Combined with another order, waiting for assignment
    ASSIGNED = "assigned"    # Assigned to a driver, waiting for pickup
    PICKED_UP = "picked_up"  # Food has been collected, en route to customer
    DELIVERED = "delivered"  # Food has been delivered to customer

class DriverState:
    """States that a driver can be in throughout their lifecycle"""
    OFFLINE = "offline"      # Not available for deliveries
    AVAILABLE = "available"  # Ready to accept assignments
    DELIVERING = "delivering"  # Actively engaged in delivery process

class PairState:
    """States that a pair can be in throughout its lifecycle"""
    CREATED = "created"      # Pair has been formed but not assigned
    ASSIGNED = "assigned"    # Pair is assigned to a driver
    COMPLETED = "completed"  # Both orders have been delivered    

class DeliveryUnitState:
    """States that a delivery unit can be in throughout its lifecycle"""
    IN_PROGRESS = "in_progress"  # Actively being delivered
    COMPLETED = "completed"      # All contained orders have been delivered   