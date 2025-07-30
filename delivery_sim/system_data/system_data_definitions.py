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
            
            # System performance analysis metrics
            'unassigned_orders_count': self.count_unassigned_orders(),
            'unassigned_pairs_count': self.count_unassigned_pairs(),
            'available_drivers_count': self.count_available_drivers(),
            'delivering_drivers_count': self.count_delivering_drivers(),
            
            # Warmup detection and supply-demand dynamics metrics
            'active_drivers': self.count_active_drivers(),
            'unassigned_delivery_entities': self.count_unassigned_delivery_entities()
        }
    
    # ===== System Performance Metrics =====
    
    def count_unassigned_orders(self):
        """Count orders waiting for assignment (not yet assigned to drivers)."""
        return len(self.repositories['order'].find_by_state(OrderState.CREATED))
    
    def count_unassigned_pairs(self):
        """Count pairs waiting for assignment (formed but not assigned)."""
        return len(self.repositories['pair'].find_by_state(PairState.CREATED))
    
    def count_available_drivers(self):
        """Count drivers available for new assignments."""
        return len(self.repositories['driver'].find_by_state(DriverState.AVAILABLE))
    
    def count_delivering_drivers(self):
        """Count drivers currently performing deliveries."""
        return len(self.repositories['driver'].find_by_state(DriverState.DELIVERING))
    
    # ===== Warmup Detection and Supply-Demand Dynamics Metrics =====
    
    def count_active_drivers(self):
        """
        Count all drivers who are actively participating in the system.
        
        This includes both available and delivering drivers, but excludes
        drivers who have logged out. This metric is KEY FOR WARMUP DETECTION
        as it shows the system's progression from zero drivers to stable
        driver population levels. Should converge to Little's Law theoretical
        average in steady state.
        
        Returns:
            int: Number of active drivers (AVAILABLE + DELIVERING)
        """
        all_drivers = self.repositories['driver'].find_all()
        active_drivers = [driver for driver in all_drivers 
                         if driver.state != DriverState.OFFLINE]
        return len(active_drivers)
    
    def count_unassigned_delivery_entities(self):
        """
        Count all delivery entities that exist in the system but are unassigned.
        
        This includes:
        - Orders in CREATED state (waiting for pairing or assignment)
        - Pairs in CREATED state (waiting for assignment)
        
        This metric shows SUPPLY-DEMAND RESPONSE DYNAMICS by revealing how
        order backlog varies in response to driver capacity fluctuations.
        Juxtaposed with 'active_drivers', it demonstrates the system's
        supply-demand balance evolution over time.
        
        Note: "delivery_entities" refers to assignable units (orders/pairs),
        not to "delivery_units" which are assignment contracts.
        
        Returns:
            int: Number of unassigned delivery entities
        """
        unassigned_orders = len(self.repositories['order'].find_by_state(OrderState.CREATED))
        unassigned_pairs = len(self.repositories['pair'].find_by_state(PairState.CREATED))
        
        return unassigned_orders + unassigned_pairs