from delivery_sim.entities.states import OrderState, DriverState, PairState
from delivery_sim.utils.logging_system import get_logger

class SystemDataDefinitions:
    """
    Defines what system metrics mean and how to calculate them.
    
    This class encapsulates all the business logic for measuring
    system state, separate from when/how often to measure.
    
    Enhanced with warmup detection metrics.
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
            dict: Complete snapshot with all system metrics including warmup detection metrics
        """
        return {
            'timestamp': timestamp,
            
            # Existing metrics for system performance analysis
            'active_orders_count': self.count_active_orders(),
            'waiting_pairs_count': self.count_waiting_pairs(),
            'total_waiting_entities': self.count_total_waiting_entities(),
            'available_drivers_count': self.count_available_drivers(),
            'delivering_drivers_count': self.count_delivering_drivers(),
            
            # NEW: Warmup detection metrics
            'active_drivers': self.count_active_drivers(),
            'active_delivery_entities': self.count_active_delivery_entities()
        }
    
    # ===== Existing System Performance Metrics =====
    
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
    
    # ===== NEW: Warmup Detection Metrics =====
    
    def count_active_drivers(self):
        """
        Count all drivers who are actively participating in the system.
        
        This includes both available and delivering drivers, but excludes
        drivers who have logged out. This metric is key for warmup detection
        as it shows the system's progression from zero drivers to stable
        driver population levels.
        
        Returns:
            int: Number of active drivers (AVAILABLE + DELIVERING)
        """
        all_drivers = self.repositories['driver'].find_all()
        active_drivers = [driver for driver in all_drivers 
                         if driver.state != DriverState.OFFLINE]
        return len(active_drivers)
    
    def count_active_delivery_entities(self):
        """
        Count all delivery entities that exist in the system but are unassigned.
        
        This includes:
        - Orders in CREATED state (waiting for pairing or assignment)
        - Pairs in CREATED state (waiting for assignment)
        
        This metric is crucial for warmup detection as it reflects the
        supply-demand balance. In steady state, this should stabilize
        around some level. If it grows indefinitely, the system may be
        in failure mode (demand >> supply).
        
        Note: "delivery_entities" refers to assignable units (orders/pairs),
        not to "delivery_units" which are assignment contracts.
        
        Returns:
            int: Number of unassigned delivery entities
        """
        unassigned_orders = len(self.repositories['order'].find_by_state(OrderState.CREATED))
        unassigned_pairs = len(self.repositories['pair'].find_by_state(PairState.CREATED))
        
        return unassigned_orders + unassigned_pairs