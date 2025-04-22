from delivery_sim.entities.states import PairState
from delivery_sim.events.pair_events import PairStateChangedEvent


class Pair:
    """
    Represents a pair of orders that will be delivered together.
    
    A pair maintains references to its constituent orders and manages
    the coordinated delivery of both orders.
    """
    
    def __init__(self, order1, order2, creation_time):
        """
        Initialize a new pair from two orders.
        
        Args:
            order1: First order in the pair
            order2: Second order in the pair
            creation_time: When this pair was created
        """
        # Core identification
        self.pair_id = f"{order1.order_id}-{order2.order_id}"
        self.order1 = order1
        self.order2 = order2
        
        # Timing information
        self.creation_time = creation_time
        self.assignment_time = None
        self.completion_time = None
        
        # State tracking
        self.state = PairState.CREATED
        self.picked_up_orders = set()  # Set of order IDs that have been picked up
        self.delivered_orders = set()  # Set of order IDs that have been delivered
        
        # Delivery planning
        self.optimal_sequence = None  # List of locations in optimal visiting order
        self.optimal_cost = None      # Total travel distance for optimal sequence
        
        # Relationship tracking
        self.delivery_unit = None
    
    def can_transition_to(self, new_state):
        """
        Check if this pair can transition to the specified state.
        
        Args:
            new_state: The state to check transition to
            
        Returns:
            bool: True if transition is valid, False otherwise
        """
        valid_transitions = {
            PairState.CREATED: [PairState.ASSIGNED],
            PairState.ASSIGNED: [PairState.COMPLETED]
        }
        
        # Special case for COMPLETED - requires both orders to be delivered
        if new_state == PairState.COMPLETED:
            return (self.state == PairState.ASSIGNED and 
                   len(self.delivered_orders) == 2)
        
        # Standard transition check
        return new_state in valid_transitions.get(self.state, [])
    
    def transition_to(self, new_state, event_dispatcher=None, env=None):
        """
        Change the pair's state with validation.
        
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
                f"Cannot transition pair {self.pair_id} from {self.state} to {new_state}"
            )
        
        # Store old state for event
        old_state = self.state
        
        # Update state and timing information
        self.state = new_state
        
        if new_state == PairState.ASSIGNED:
            self.assignment_time = env.now if env else None
        elif new_state == PairState.COMPLETED:
            self.completion_time = env.now if env else None
        
        # Dispatch event if dispatcher provided
        if event_dispatcher and env:
            event_dispatcher.dispatch(PairStateChangedEvent(
                timestamp=env.now,
                pair_id=self.pair_id,
                old_state=old_state,
                new_state=new_state
            ))
        
        return True
    
    def record_order_pickup(self, order_id):
        """
        Record that an order in this pair has been picked up.
        
        Args:
            order_id: ID of the order that was picked up
            
        Returns:
            bool: True if this updated the pair state, False otherwise
        """
        if order_id not in [self.order1.order_id, self.order2.order_id]:
            return False
        
        self.picked_up_orders.add(order_id)
        return True
    
    def record_order_delivery(self, order_id):
        """
        Record that an order in this pair has been delivered.
        
        Args:
            order_id: ID of the order that was delivered
            
        Returns:
            bool: True if this updated the pair state, False otherwise
        """
        if order_id not in [self.order1.order_id, self.order2.order_id]:
            return False
        
        self.delivered_orders.add(order_id)
        return len(self.delivered_orders) == 2  # Return True if pair is now complete