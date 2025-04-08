class OrderState:
    """States that an order can be in throughout its lifecycle"""
    CREATED = "created"      # Just entered the system
    PAIRED = "paired"        # Combined with another order
    ASSIGNED = "assigned"    # Assigned to a driver
    PICKED_UP = "picked_up"  # Driver has collected the order
    DELIVERED = "delivered"  # Order has been delivered to customer

class DriverState:
    """States that a driver can be in throughout their lifecycle"""
    OFFLINE = "offline"      # Not available for deliveries
    AVAILABLE = "available"  # Ready to accept assignments
    ASSIGNED = "assigned"    # Assigned to a delivery
    PICKING_UP = "picking_up"  # En route to restaurant
    DELIVERING = "delivering"  # En route to customer