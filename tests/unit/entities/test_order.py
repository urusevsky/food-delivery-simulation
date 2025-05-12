# tests/unit/entities/test_order.py
import pytest
import simpy
from delivery_sim.entities.order import Order
from delivery_sim.entities.states import OrderState
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.events.order_events import OrderStateChangedEvent

# Group 1: Testing Order creation and basic properties
def test_order_creation():
    """
    Test that an Order can be created with proper initialization of all attributes.
    This verifies our constructor works correctly.
    """
    # SETUP - prepare the test objects and initial conditions
    order_id = "O123"
    restaurant_location = [0, 0]
    customer_location = [2, 3]
    arrival_time = 100
    
    # EXECUTE - perform the action being tested
    order = Order(order_id, restaurant_location, customer_location, arrival_time)
    
    # VERIFY - check that the expected outcomes occurred
    assert order.order_id == order_id, "Order ID should match the provided value"
    assert order.restaurant_location == restaurant_location, "Restaurant location should match the provided value"
    assert order.customer_location == customer_location, "Customer location should match the provided value"
    assert order.arrival_time == arrival_time, "Arrival time should match the provided value"
    assert order.state == OrderState.CREATED, "Initial state should be CREATED"
    assert order.pair is None, "Initial pair should be None"
    assert order.delivery_unit is None, "Initial delivery_unit should be None"

# Group 2: Testing state transitions (valid cases)
def test_all_valid_state_transitions():
    """
    Test all possible valid state transition paths to ensure complete coverage.
    """
    # Path 1: Order gets paired
    order1 = Order("O101", [0, 0], [1, 1], 0)
    
    # CREATED -> PAIRED
    order1.transition_to(OrderState.PAIRED)
    assert order1.state == OrderState.PAIRED
    
    # PAIRED -> ASSIGNED
    order1.transition_to(OrderState.ASSIGNED)
    assert order1.state == OrderState.ASSIGNED
    
    # Continue with regular flow...
    order1.transition_to(OrderState.PICKED_UP)
    order1.transition_to(OrderState.DELIVERED)
    
    # Path 2: Order stays single (direct assignment)
    order2 = Order("O102", [0, 0], [1, 1], 0)
    
    # CREATED -> ASSIGNED (skipping PAIRED)
    order2.transition_to(OrderState.ASSIGNED)
    assert order2.state == OrderState.ASSIGNED
    
    # Continue with regular flow...
    order2.transition_to(OrderState.PICKED_UP)
    order2.transition_to(OrderState.DELIVERED)

# Group 3: Testing invalid state transitions
def test_invalid_direct_transition_to_delivered():
    """
    Test that an order cannot transition directly from CREATED to DELIVERED.
    This verifies our state validation prevents invalid transitions.
    """
    # SETUP
    order = Order("O123", [0, 0], [1, 1], 0)
    
    # EXECUTE & VERIFY - Check that an invalid transition raises ValueError
    with pytest.raises(ValueError):
        # This should fail because you can't go directly from CREATED to DELIVERED
        order.transition_to(OrderState.DELIVERED)
    
    # Also verify the state hasn't changed
    assert order.state == OrderState.CREATED, "State should remain unchanged after failed transition"

def test_invalid_backward_transition():
    """
    Test that an order cannot transition backward in its lifecycle.
    This verifies our state validation prevents backward movement.
    """
    # SETUP - Create an order and move it to PAIRED state
    order = Order("O123", [0, 0], [1, 1], 0)
    order.transition_to(OrderState.PAIRED)
    
    # EXECUTE & VERIFY - Check that going back to CREATED raises ValueError
    with pytest.raises(ValueError):
        order.transition_to(OrderState.CREATED)
    
    # Verify the state hasn't changed
    assert order.state == OrderState.PAIRED, "State should remain unchanged after failed transition"

# Group 4: Testing string representation
def test_order_string_representation():
    """
    Test that the string representation of an Order shows its key information.
    This verifies our __str__ method works correctly.
    """
    # SETUP
    order = Order("O123", [0, 0], [1, 1], 0)
    
    # EXECUTE
    string_repr = str(order)
    
    # VERIFY
    assert "Order(id=O123" in string_repr, "String representation should include the order ID"
    assert f"state={order.state}" in string_repr, "String representation should include the state"

def test_order_state_change_dispatches_event():
    """Test that order state changes generate events when a dispatcher is provided."""
    # Setup
    env = simpy.Environment()
    dispatcher = EventDispatcher()
    order = Order("O123", [0, 0], [2, 3], 100)  # ID, restaurant_loc, customer_loc, arrival_time
    
    # Track received events
    received_events = []
    def test_handler(event):
        received_events.append(event)
    
    # Register handler
    dispatcher.register(OrderStateChangedEvent, test_handler)
    
    # Change state with dispatcher and env
    order.transition_to(OrderState.PAIRED, dispatcher, env)
    
    # Verify event was dispatched with correct data
    assert len(received_events) == 1
    event = received_events[0]
    assert isinstance(event, OrderStateChangedEvent)
    assert event.order_id == "O123"
    assert event.old_state == OrderState.CREATED
    assert event.new_state == OrderState.PAIRED
    assert event.timestamp == env.now