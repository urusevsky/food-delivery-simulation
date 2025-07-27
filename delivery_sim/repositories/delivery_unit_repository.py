from delivery_sim.entities.states import DeliveryUnitState


class DeliveryUnitRepository:
    def __init__(self):
        self.delivery_units = {}  # Maps unit_id to DeliveryUnit objects
    
    def add(self, delivery_unit):
        self.delivery_units[delivery_unit.unit_id] = delivery_unit
    
    def find_by_id(self, unit_id):
        return self.delivery_units.get(unit_id)
    
    def find_all(self):
        return list(self.delivery_units.values())
    
    def find_by_state(self, state):
        return [unit for unit in self.delivery_units.values() 
                if unit.state == state]
    
    def find_active_deliveries(self):
        return self.find_by_state(DeliveryUnitState.IN_PROGRESS)
    
    def find_by_driver_id(self, driver_id):
        return [unit for unit in self.delivery_units.values() 
                if unit.driver.driver_id == driver_id]
                