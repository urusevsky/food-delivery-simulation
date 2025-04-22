from delivery_sim.events.base_events import Event

class DeliveryUnitEvent(Event):
    """Base class for all delivery unit events."""
    def __init__(self, timestamp, delivery_unit_id):
        super().__init__(timestamp)
        self.delivery_unit_id = delivery_unit_id

class DeliveryAssignedEvent(DeliveryUnitEvent):
    """Event for when a delivery entity is assigned to a driver."""
    def __init__(self, timestamp, delivery_unit_id, entity_type, entity_id, driver_id):
        super().__init__(timestamp, delivery_unit_id)
        self.entity_type = entity_type  # "order" or "pair"
        self.entity_id = entity_id
        self.driver_id = driver_id
        
class DeliveryCompletedEvent(DeliveryUnitEvent):
    """Event for when a delivery is fully completed."""
    def __init__(self, timestamp, delivery_unit_id, driver_id):
        super().__init__(timestamp, delivery_unit_id)
        self.driver_id = driver_id