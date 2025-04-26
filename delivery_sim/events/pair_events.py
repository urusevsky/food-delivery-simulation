from delivery_sim.events.base_events import Event

class PairEvent(Event):
    """Base class for all pair-related events."""
    def __init__(self, timestamp, pair_id):
        super().__init__(timestamp)
        self.pair_id = pair_id

class PairCreatedEvent(PairEvent):
    """Event for when a new pair is formed."""
    def __init__(self, timestamp, pair_id, order1_id, order2_id):
        super().__init__(timestamp, pair_id)
        self.order1_id = order1_id
        self.order2_id = order2_id

class PairingFailedEvent(Event):
    """Event for when an order failed to find a suitable pairing match."""
    def __init__(self, timestamp, order_id):
        super().__init__(timestamp)
        self.order_id = order_id        

class PairAssignedEvent(PairEvent):
    """Event for when a pair is assigned to a driver."""
    def __init__(self, timestamp, pair_id, driver_id, delivery_unit_id):
        super().__init__(timestamp, pair_id)
        self.driver_id = driver_id
        self.delivery_unit_id = delivery_unit_id

class PairCompletedEvent(PairEvent):
    """Event for when both orders in a pair have been delivered."""
    def __init__(self, timestamp, pair_id):
        super().__init__(timestamp, pair_id)

class PairStateChangedEvent(PairEvent):
    """Technical event for tracking all pair state transitions."""
    def __init__(self, timestamp, pair_id, old_state, new_state):
        super().__init__(timestamp, pair_id)
        self.old_state = old_state
        self.new_state = new_state