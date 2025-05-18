from delivery_sim.events.base_events import Event

class DeliveryUnitEvent(Event):
    """Base class for all delivery unit events."""
    def __init__(self, timestamp, delivery_unit_id):
        super().__init__(timestamp)
        self.delivery_unit_id = delivery_unit_id

class DeliveryUnitAssignedEvent(DeliveryUnitEvent):
    """Event for when a delivery entity is assigned to a driver."""
    def __init__(self, timestamp, delivery_unit_id, entity_type, entity_id, driver_id):
        super().__init__(timestamp, delivery_unit_id)
        self.entity_type = entity_type  # EntityType.ORDER or EntityType.PAIR
        self.entity_id = entity_id
        self.driver_id = driver_id
        
class DeliveryUnitCompletedEvent(DeliveryUnitEvent):
    """Event for when a delivery is fully completed."""
    def __init__(self, timestamp, delivery_unit_id, driver_id):
        super().__init__(timestamp, delivery_unit_id)
        self.driver_id = driver_id

class DeliveryUnitStateChangedEvent(DeliveryUnitEvent):
    """Technical event for tracking delivery unit state transitions."""
    def __init__(self, timestamp, delivery_unit_id, old_state, new_state):
        super().__init__(timestamp, delivery_unit_id)
        self.old_state = old_state
        self.new_state = new_state        