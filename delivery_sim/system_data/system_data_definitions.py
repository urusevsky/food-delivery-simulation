from delivery_sim.entities.states import OrderState, DriverState, PairState
from delivery_sim.utils.logging_system import get_logger

class SystemDataDefinitions:
    """
    Defines what system metrics mean and how to calculate them.
    
    This class encapsulates all the business logic for measuring
    system state, separate from when/how often to measure.
    """
    
    def __init__(self, repositories):
        """
        Initialize with entity repositories for state queries.
        
        Args:
            repositories: Dict with 'order', 'driver', 'pair', 'delivery_unit' repositories
        """
        self.repositories = repositories
        self.logger = get_logger("system_data.definitions")
    
    def create_snapshot_data(self, timestamp):
        """
        Create complete snapshot data dictionary.
        
        Args:
            timestamp: Current simulation time
            
        Returns:
            dict: Complete snapshot with all system metrics
        """
        return {
            'timestamp': timestamp,
            'active_orders_count': self.count_active_orders(),
            'waiting_pairs_count': self.count_waiting_pairs(),
            'total_waiting_entities': self.count_total_waiting_entities(),
            'available_drivers_count': self.count_available_drivers(),
            'delivering_drivers_count': self.count_delivering_drivers()
        }
    
    # Individual metric calculation methods
    def count_active_orders(self):
        """Count orders waiting for assignment (not yet assigned to drivers)."""
        return len(self.repositories['order'].find_by_state(OrderState.CREATED))
    
    def count_waiting_pairs(self):
        """Count pairs waiting for assignment (formed but not assigned)."""
        return len(self.repositories['pair'].find_by_state(PairState.CREATED))
    
    def count_total_waiting_entities(self):
        """Count all entities (orders + pairs) waiting for driver assignment."""
        return self.count_active_orders() + self.count_waiting_pairs()
    
    def count_available_drivers(self):
        """Count drivers available for new assignments."""
        return len(self.repositories['driver'].find_by_state(DriverState.AVAILABLE))
    
    def count_delivering_drivers(self):
        """Count drivers currently performing deliveries."""
        return len(self.repositories['driver'].find_by_state(DriverState.DELIVERING))
    
