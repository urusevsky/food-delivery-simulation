from delivery_sim.entities.states import DeliveryUnitState

class DeliveryUnit:
    """
    Represents the assignment contract between a driver and a delivery entity.
    
    DeliveryUnit captures the relationship created when a driver is assigned
    to deliver either a single order or a pair of orders.
    """
    
    def __init__(self, delivery_entity, driver, assignment_time):
        """
        Initialize a new delivery unit.
        
        Args:
            delivery_entity: The Order or Pair being delivered
            driver: The Driver performing the delivery
            assignment_time: When this assignment was made
        """
        # Core relationships
        self.delivery_entity = delivery_entity
        self.driver = driver
        
        # Generate ID based on entity type
        entity_type = "P" if hasattr(delivery_entity, 'pair_id') else "O"
        entity_id = (delivery_entity.pair_id if entity_type == "P" 
                    else delivery_entity.order_id)
        self.unit_id = f"DU-{entity_type}-{entity_id}-{driver.driver_id}"
        
        # State and timing
        self.state = DeliveryUnitState.IN_PROGRESS
        self.assignment_time = assignment_time
        self.completion_time = None
        
        # Assignment decision information
        self.assignment_path = None  # "immediate" or "periodic"
        self.assignment_costs = {
            "base_cost": None,        # Raw distance cost
            "throughput_factor": None,  # Lambda value used
            "throughput_discount": None,  # Discount for delivering multiple orders
            "age_factor": None,       # Mu value used
            "age_discount": None,     # Discount for waiting time
            "adjusted_cost": None     # Final cost used in decision
        }
    
    def can_transition_to(self, new_state):
        """
        Check if this delivery unit can transition to the specified state.
        
        Args:
            new_state: The state to check transition to
            
        Returns:
            bool: True if transition is valid, False otherwise
        """
        valid_transitions = {
            DeliveryUnitState.IN_PROGRESS: [DeliveryUnitState.COMPLETED]
        }
        
        return new_state in valid_transitions.get(self.state, [])
    
    def transition_to(self, new_state, event_dispatcher=None, env=None):
        """
        Change the delivery unit's state with validation.
        
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
                f"Cannot transition delivery unit {self.unit_id} from {self.state} to {new_state}"
            )
        
        # Store old state for event
        old_state = self.state
        
        # Update state and timing information
        self.state = new_state
        
        if new_state == DeliveryUnitState.COMPLETED:
            self.completion_time = env.now if env else None
        
        # Dispatch event if dispatcher provided
        if event_dispatcher and env:
            event_dispatcher.dispatch(DeliveryUnitStateChangedEvent(
                timestamp=env.now,
                delivery_unit_id=self.unit_id,
                old_state=old_state,
                new_state=new_state
            ))
        
        return True