from delivery_sim.entities.states import OrderState
from delivery_sim.events.order_events import OrderStateChangedEvent

class Order:
    """
    Represents a food delivery order in the system.
    
    This is a fundamental entity in the simulation that contains
    all necessary information about an individual order.
    """
    
    def __init__(self, order_id, restaurant_location, customer_location, arrival_time):
        """
        Initialize a new order with its basic properties.
        
        Args:
            order_id: Unique identifier for this order
            restaurant_location: (x, y) coordinates of the restaurant
            customer_location: (x, y) coordinates of the customer
            arrival_time: When this order entered the system
        """
        # Basic properties
        self.order_id = order_id
        self.restaurant_location = restaurant_location
        self.customer_location = customer_location
        self.arrival_time = arrival_time
        
        # State and relationships (initially None)
        self.state = OrderState.CREATED
        self.pair = None  # Reference to a Pair if this order becomes paired
        self.delivery_unit = None  # Reference to DeliveryUnit once assigned
        
        # Timing information for lifecycle events
        self.pair_time = None
        self.assignment_time = None
        self.pickup_time = None
        self.delivery_time = None

    def can_transition_to(self, new_state):
        """
        Check if this order can transition to the specified state.
        
        Args:
            new_state: The state to check transition to
            
        Returns:
            bool: True if transition is valid, False otherwise
        """
        valid_transitions = {
            OrderState.CREATED: [OrderState.PAIRED, OrderState.ASSIGNED],
            OrderState.PAIRED: [OrderState.ASSIGNED],
            OrderState.ASSIGNED: [OrderState.PICKED_UP],
            OrderState.PICKED_UP: [OrderState.DELIVERED]
        }
        
        return new_state in valid_transitions.get(self.state, [])

    def transition_to(self, new_state, event_dispatcher=None, env=None):
        """
        Change the order's state with validation.
        
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
                f"Cannot transition order {self.order_id} from {self.state} to {new_state}"
            )
        
        # Store old state for event
        old_state = self.state
        
        # Update state
        self.state = new_state
        
        # Update timing information based on new state
        if env:
            if new_state == OrderState.PAIRED:
                self.pair_time = env.now
            elif new_state == OrderState.ASSIGNED:
                self.assignment_time = env.now
            elif new_state == OrderState.PICKED_UP:
                self.pickup_time = env.now
            elif new_state == OrderState.DELIVERED:
                self.delivery_time = env.now
        
        # Dispatch event if dispatcher provided
        if event_dispatcher and env:
            event_dispatcher.dispatch(OrderStateChangedEvent(
                timestamp=env.now,
                order_id=self.order_id,
                old_state=old_state,
                new_state=new_state
            ))
        
        return True
    
    def __str__(self):
        """String representation of the order"""
        return f"Order(id={self.order_id}, state={self.state})"