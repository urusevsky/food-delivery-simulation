from .base_events import Event

class OrderEvent(Event):
    """Base class for all order-related events."""
    
    def __init__(self, timestamp, order_id):
        """
        Initialize an order event.
        
        Args:
            timestamp: When this event occurred
            order_id: ID of the order this event relates to
        """
        super().__init__(timestamp)
        self.order_id = order_id

class OrderCreatedEvent(OrderEvent):
    """Event for when a new order enters the system."""
    
    def __init__(self, timestamp, order_id, restaurant_location, customer_location):
        """
        Initialize an order created event.
        
        Args:
            timestamp: When this event occurred
            order_id: ID of the newly created order
            restaurant_location: Location of the restaurant
            customer_location: Location of the customer
        """
        super().__init__(timestamp, order_id)
        self.restaurant_location = restaurant_location
        self.customer_location = customer_location

class OrderStateChangedEvent(OrderEvent):
    """Event for when an order's state changes."""
    
    def __init__(self, timestamp, order_id, old_state, new_state):
        """
        Initialize an order state changed event.
        
        Args:
            timestamp: When this event occurred
            order_id: ID of the order whose state changed
            old_state: The state the order was in before the change
            new_state: The state the order is in after the change
        """
        super().__init__(timestamp, order_id)
        self.old_state = old_state
        self.new_state = new_state