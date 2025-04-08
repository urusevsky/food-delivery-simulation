from .states import DriverState

class Driver:
    """
    Represents a delivery driver in the system.
    
    Drivers handle the actual delivery of orders and have their
    own state lifecycle as they move through the system.
    """
    
    def __init__(self, driver_id, initial_location, login_time, service_duration):
        """
        Initialize a new driver with their properties.
        
        Args:
            driver_id: Unique identifier for this driver
            initial_location: (x, y) coordinates where driver starts
            login_time: When this driver became available
            service_duration: How long driver intends to offer service
        """
        # Basic properties
        self.driver_id = driver_id
        self.location = initial_location
        self.login_time = login_time
        self.service_duration = service_duration  # From distribution
        self.intended_logout_time = login_time + service_duration
        
        # State and assignment tracking
        self.state = DriverState.AVAILABLE
        self.current_delivery = None  # Reference to assigned DeliveryUnit
        self.last_state_change_time = login_time
        
    def transition_to(self, new_state, event_dispatcher=None):
        """
        Change the driver's state with validation.
        
        Args:
            new_state: The new state to transition to
            event_dispatcher: Optional event dispatcher to notify about state change
            
        Returns:
            True if transition was successful
            
        Raises:
            ValueError: If transition is invalid
        """
        # Define valid transitions from each state
        valid_transitions = {
            DriverState.OFFLINE: [DriverState.AVAILABLE],
            DriverState.AVAILABLE: [DriverState.ASSIGNED, DriverState.OFFLINE],
            DriverState.ASSIGNED: [DriverState.PICKING_UP],
            DriverState.PICKING_UP: [DriverState.DELIVERING],
            DriverState.DELIVERING: [DriverState.AVAILABLE, DriverState.OFFLINE]
        }
        
        # Check if transition is valid
        if self.state not in valid_transitions or new_state not in valid_transitions.get(self.state, []):
            raise ValueError(
                f"Cannot transition driver {self.driver_id} from {self.state} to {new_state}"
            )
        
        # Store old state for event
        old_state = self.state
        
        # Update state
        self.state = new_state
        # We'll update this with actual timestamp when we integrate with SimPy
        # self.last_state_change_time = SimulationClock.now()
        
        # We'll handle event dispatching later when we integrate events
        # if event_dispatcher:
        #     event_dispatcher.dispatch(DriverStateChangedEvent(...))
        
        return True
    
    def update_location(self, new_location):
        """
        Update the driver's current location.
        
        Args:
            new_location: (x, y) coordinates of new location
        """
        self.location = new_location
        
    def can_logout(self):
        """
        Check if driver can log out based on current state and assignments.
        
        Returns:
            True if the driver can log out, False otherwise
        """
        # A driver can only log out if they're available (no active deliveries)
        return self.state == DriverState.AVAILABLE
    
    def __str__(self):
        """String representation of the driver"""
        return f"Driver(id={self.driver_id}, state={self.state}, location={self.location})"