from delivery_sim.entities.states import DriverState
from delivery_sim.events.driver_events import DriverStateChangedEvent

class Driver:
    """
    Represents a delivery driver in the system.
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
        self.service_duration = service_duration
        self.intended_logout_time = login_time + service_duration
        
        # State and assignment
        self.state = DriverState.AVAILABLE
        self.current_delivery_unit = None
        self.completed_deliveries = []
        
        # Timing information for state changes
        self.last_state_change_time = login_time
        self.actual_logout_time = None
    
    def can_transition_to(self, new_state):
        """
        Check if this driver can transition to the specified state.
        
        Args:
            new_state: The state to check transition to
            
        Returns:
            bool: True if transition is valid, False otherwise
        """
        valid_transitions = {
            DriverState.OFFLINE: [DriverState.AVAILABLE],
            DriverState.AVAILABLE: [DriverState.DELIVERING, DriverState.OFFLINE],
            DriverState.DELIVERING: [DriverState.AVAILABLE, DriverState.OFFLINE]
        }
        
        return new_state in valid_transitions.get(self.state, [])
    
    def transition_to(self, new_state, event_dispatcher=None, env=None):
        """
        Change the driver's state with validation.
        
        Args:
            new_state: The new state to transition to
            event_dispatcher: Optional event dispatcher for events
            env: SimPy environment for current time
            
        Returns:
            bool: True if transition succeeded, False otherwise
            
        Raises:
            ValueError: If transition is invalid
        """
        if not self.can_transition_to(new_state):
            raise ValueError(
                f"Cannot transition driver {self.driver_id} from {self.state} to {new_state}"
            )
        
        # Store old state for event
        old_state = self.state
        
        # Update state
        self.state = new_state
        
        # Update timing information
        current_time = env.now if env else None
        if current_time:
            self.last_state_change_time = current_time
            
            # Record logout time if transitioning to OFFLINE
            if new_state == DriverState.OFFLINE:
                self.actual_logout_time = current_time
        
        # Dispatch event if dispatcher provided
        if event_dispatcher and env:
            event_dispatcher.dispatch(DriverStateChangedEvent(
                timestamp=env.now,
                driver_id=self.driver_id,
                old_state=old_state,
                new_state=new_state
            ))
        
        return True
    
    def update_location(self, new_location):
        """
        Update the driver's current location.
        
        Args:
            new_location: (x, y) coordinates of new location
        """
        self.location = new_location
    
    def __str__(self):
        """String representation of the driver"""
        return f"Driver(id={self.driver_id}, state={self.state}, location={self.location})"