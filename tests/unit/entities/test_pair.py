
import pytest
import simpy
from delivery_sim.entities.pair import Pair
from delivery_sim.entities.order import Order
from delivery_sim.entities.states import PairState, OrderState
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.events.pair_events import PairStateChangedEvent

def test_pair_initialization():
    """Test that a Pair is properly initialized with correct attributes."""
    # Create two orders
    order1 = Order("1", [0, 0], [1, 1], 100)
    order2 = Order("2", [0, 0], [2, 2], 105)
    creation_time = 110
    
    # Create pair
    pair = Pair(order1, order2, creation_time)
    
    # Verify basic properties
    assert pair.pair_id == "1-2"
    assert pair.order1 is order1
    assert pair.order2 is order2
    assert pair.creation_time == creation_time
    assert pair.assignment_time is None
    assert pair.completion_time is None
    assert pair.state == PairState.CREATED
    assert len(pair.picked_up_orders) == 0
    assert len(pair.delivered_orders) == 0
    assert pair.optimal_sequence is None
    assert pair.optimal_cost is None
    assert pair.delivery_unit is None

def test_valid_state_transitions():
    """Test that valid state transitions work properly."""
    # Setup
    order1 = Order("1", [0, 0], [1, 1], 100)
    order2 = Order("2", [0, 0], [2, 2], 105)
    env = simpy.Environment()
    
    pair = Pair(order1, order2, env.now)
    
    # Test CREATED -> ASSIGNED
    pair.transition_to(PairState.ASSIGNED, None, env)
    assert pair.state == PairState.ASSIGNED
    assert pair.assignment_time == env.now
    
    # Record deliveries to enable completion transition
    pair.record_order_delivery("1")
    pair.record_order_delivery("2")
    
    # Test ASSIGNED -> COMPLETED
    pair.transition_to(PairState.COMPLETED, None, env)
    assert pair.state == PairState.COMPLETED
    assert pair.completion_time == env.now

def test_invalid_state_transitions():
    """Test that invalid state transitions raise appropriate errors."""
    # Setup
    order1 = Order("1", [0, 0], [1, 1], 100)
    order2 = Order("2", [0, 0], [2, 2], 105)
    env = simpy.Environment()
    
    pair = Pair(order1, order2, env.now)
    
    # Test invalid CREATED -> COMPLETED (skipping ASSIGNED)
    with pytest.raises(ValueError):
        pair.transition_to(PairState.COMPLETED, None, env)
    
    # Move to ASSIGNED for further tests
    pair.transition_to(PairState.ASSIGNED, None, env)
    
    # Test ASSIGNED -> COMPLETED without all orders delivered
    # This should fail because completion requires both orders to be delivered
    with pytest.raises(ValueError):
        pair.transition_to(PairState.COMPLETED, None, env)
    
    # Test invalid backward transition
    pair.record_order_delivery("1")
    pair.record_order_delivery("2")
    pair.transition_to(PairState.COMPLETED, None, env)
    
    # Cannot go back from COMPLETED to ASSIGNED
    with pytest.raises(ValueError):
        pair.transition_to(PairState.ASSIGNED, None, env)

def test_event_dispatch_on_state_change():
    """Test that state changes generate events when a dispatcher is provided."""
    # Setup
    order1 = Order("1", [0, 0], [1, 1], 100)
    order2 = Order("2", [0, 0], [2, 2], 105)
    env = simpy.Environment()
    dispatcher = EventDispatcher()
    
    # Track received events
    received_events = []
    def test_handler(event):
        received_events.append(event)
    
    # Register handler
    dispatcher.register(PairStateChangedEvent, test_handler)
    
    # Create pair
    pair = Pair(order1, order2, env.now)
    
    # Change state with dispatcher
    pair.transition_to(PairState.ASSIGNED, dispatcher, env)
    
    # Verify event was dispatched with correct data
    assert len(received_events) == 1
    event = received_events[0]
    assert isinstance(event, PairStateChangedEvent)
    assert event.pair_id == pair.pair_id
    assert event.old_state == PairState.CREATED
    assert event.new_state == PairState.ASSIGNED
    assert event.timestamp == env.now

def test_order_pickup_tracking():
    """Test that order pickups are correctly tracked."""
    # Setup
    order1 = Order("1", [0, 0], [1, 1], 100)
    order2 = Order("2", [0, 0], [2, 2], 105)
    pair = Pair(order1, order2, 110)
    
    # Record pickup for first order
    result = pair.record_order_pickup("1")
    assert result is True
    assert len(pair.picked_up_orders) == 1
    assert "1" in pair.picked_up_orders
    
    # Record pickup for second order
    result = pair.record_order_pickup("2")
    assert result is True
    assert len(pair.picked_up_orders) == 2
    assert "2" in pair.picked_up_orders
    
    # Try to record pickup for non-existent order
    result = pair.record_order_pickup("3")
    assert result is False
    assert len(pair.picked_up_orders) == 2

def test_order_delivery_tracking():
    """Test that order deliveries are correctly tracked and completion is detected."""
    # Setup
    order1 = Order("1", [0, 0], [1, 1], 100)
    order2 = Order("2", [0, 0], [2, 2], 105)
    pair = Pair(order1, order2, 110)
    
    # Record delivery for first order
    result = pair.record_order_delivery("1")
    assert result is False  # Not complete yet
    assert len(pair.delivered_orders) == 1
    assert "1" in pair.delivered_orders
    
    # Record delivery for second order
    result = pair.record_order_delivery("2")
    assert result is True  # Now complete
    assert len(pair.delivered_orders) == 2
    assert "2" in pair.delivered_orders
    
    # Try to record delivery for non-existent order
    result = pair.record_order_delivery("3")
    assert result is False
    assert len(pair.delivered_orders) == 2