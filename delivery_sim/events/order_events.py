from delivery_sim.events.base_events import Event

class OrderEvent(Event):
    """Base class for all order-related events."""
    def __init__(self, timestamp, order_id):
        super().__init__(timestamp)
        self.order_id = order_id

class OrderCreatedEvent(OrderEvent):
    """Event for when a new order enters the system."""
    def __init__(self, timestamp, order_id, restaurant_id, restaurant_location, customer_location):
        super().__init__(timestamp, order_id)
        self.restaurant_id = restaurant_id
        self.restaurant_location = restaurant_location
        self.customer_location = customer_location

class OrderPairedEvent(OrderEvent):
    """Event for when an order is paired with another order."""
    def __init__(self, timestamp, order_id, pair_id, paired_with_order_id):
        super().__init__(timestamp, order_id)
        self.pair_id = pair_id
        self.paired_with_order_id = paired_with_order_id

class OrderAssignedEvent(OrderEvent):
    """Event for when an order is assigned to a driver."""
    def __init__(self, timestamp, order_id, driver_id, delivery_unit_id):
        super().__init__(timestamp, order_id)
        self.driver_id = driver_id
        self.delivery_unit_id = delivery_unit_id

class OrderPickedUpEvent(OrderEvent):
    """Event for when a driver picks up food from the restaurant."""
    def __init__(self, timestamp, order_id, driver_id):
        super().__init__(timestamp, order_id)
        self.driver_id = driver_id

class OrderDeliveredEvent(OrderEvent):
    """Event for when a driver delivers food to the customer."""
    def __init__(self, timestamp, order_id, driver_id):
        super().__init__(timestamp, order_id)
        self.driver_id = driver_id

class OrderStateChangedEvent(OrderEvent):
    """Technical event for tracking all order state transitions."""
    def __init__(self, timestamp, order_id, old_state, new_state):
        super().__init__(timestamp, order_id)
        self.old_state = old_state
        self.new_state = new_state