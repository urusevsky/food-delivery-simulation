# tests/unit/entities/test_delivery_unit.py
"""
Updated tests for DeliveryUnit entity with priority scoring system.

Changes from adjusted cost system:
- assignment_costs → assignment_scores  
- Updated expected dictionary keys to match priority scoring components
- Updated test values to reflect 0-1 score ranges and 0-100 priority scores
"""

import pytest
import simpy
from delivery_sim.entities.delivery_unit import DeliveryUnit
from delivery_sim.entities.states import DeliveryUnitState
from delivery_sim.entities.order import Order
from delivery_sim.entities.pair import Pair
from delivery_sim.entities.driver import Driver
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.events.delivery_unit_events import DeliveryUnitStateChangedEvent
from delivery_sim.utils.entity_type_utils import EntityType

# Test Group 1: Testing DeliveryUnit initialization for single orders
def test_delivery_unit_creation_with_single_order():
    """
    Test that a DeliveryUnit can be created for a single order.
    This verifies the constructor works correctly when dealing with an Order entity.
    """
    # ARRANGE - Create the entities we need
    order = Order("O123", [0, 0], [2, 3], 100)
    driver = Driver("D1", [0, 0], 100, 120)
    assignment_time = 110
    
    # ACT - Create the delivery unit
    delivery_unit = DeliveryUnit(order, driver, assignment_time)
    
    # ASSERT - Verify all properties are initialized correctly
    assert delivery_unit.delivery_entity is order, "DeliveryUnit should reference the order"
    assert delivery_unit.driver is driver, "DeliveryUnit should reference the driver"
    assert delivery_unit.unit_id == "DU-O123-D1", "Unit ID should follow pattern DU-{order_id}-{driver_id}"
    assert delivery_unit.state == DeliveryUnitState.IN_PROGRESS, "Initial state should be IN_PROGRESS"
    assert delivery_unit.assignment_time == assignment_time, "Assignment time should match provided value"
    assert delivery_unit.completion_time is None, "Completion time should be None initially"
    assert delivery_unit.entity_type == EntityType.DELIVERY_UNIT
    
    # Verify assignment decision information is initialized (priority scoring system)
    assert delivery_unit.assignment_path is None
    assert isinstance(delivery_unit.assignment_scores, dict)
    assert all(key in delivery_unit.assignment_scores for key in [
        "distance_score", "throughput_score", "fairness_score", 
        "combined_score_0_1", "priority_score_0_100", "total_distance",
        "num_orders", "assignment_delay_minutes"
    ])
    assert all(value is None for value in delivery_unit.assignment_scores.values())

# Test Group 2: Testing DeliveryUnit initialization for pairs
def test_delivery_unit_creation_with_pair():
    """
    Test that a DeliveryUnit can be created for a pair of orders.
    This verifies the constructor correctly handles the Pair entity type.
    """
    # ARRANGE - Create the entities we need
    order1 = Order("O101", [0, 0], [1, 1], 100)
    order2 = Order("O102", [0, 0], [2, 2], 105)
    pair = Pair(order1, order2, 110)
    driver = Driver("D2", [0, 0], 100, 120)
    assignment_time = 115
    
    # ACT - Create the delivery unit
    delivery_unit = DeliveryUnit(pair, driver, assignment_time)
    
    # ASSERT - Verify the unit ID follows the correct pattern for pairs
    assert delivery_unit.delivery_entity is pair, "DeliveryUnit should reference the pair"
    assert delivery_unit.driver is driver, "DeliveryUnit should reference the driver"
    assert delivery_unit.unit_id == "DU-P-O101_O102-D2", "Unit ID should follow pattern DU-{pair_id}-{driver_id}"
    assert delivery_unit.state == DeliveryUnitState.IN_PROGRESS, "Initial state should be IN_PROGRESS"
    assert delivery_unit.assignment_time == assignment_time, "Assignment time should match provided value"

# Test Group 3: Testing state transitions
def test_valid_state_transition_to_completed():
    """
    Test that a DeliveryUnit can transition from IN_PROGRESS to COMPLETED.
    This is the primary state transition in the delivery unit lifecycle.
    """
    # ARRANGE
    order = Order("O123", [0, 0], [2, 3], 100)
    driver = Driver("D1", [0, 0], 100, 120)
    env = simpy.Environment()
    delivery_unit = DeliveryUnit(order, driver, env.now)
    
    # Move time forward to simulate delivery completion
    env.run(until=10)
    
    # ACT - Transition to completed state
    delivery_unit.transition_to(DeliveryUnitState.COMPLETED, None, env)
    
    # ASSERT - Verify the state change and timing update
    assert delivery_unit.state == DeliveryUnitState.COMPLETED, "State should be COMPLETED"
    assert delivery_unit.completion_time == env.now, "Completion time should be set to current time"

def test_invalid_state_transition():
    """
    Test that invalid state transitions are properly rejected.
    The only valid transition is IN_PROGRESS -> COMPLETED, so test other attempts.
    """
    # ARRANGE
    order = Order("O123", [0, 0], [2, 3], 100)
    driver = Driver("D1", [0, 0], 100, 120)
    delivery_unit = DeliveryUnit(order, driver, 110)
    
    # ACT & ASSERT - Try to transition from IN_PROGRESS to IN_PROGRESS (no-op)
    with pytest.raises(ValueError):
        delivery_unit.transition_to(DeliveryUnitState.IN_PROGRESS)
    
    # Verify state hasn't changed
    assert delivery_unit.state == DeliveryUnitState.IN_PROGRESS

def test_cannot_transition_from_completed():
    """
    Test that once COMPLETED, a delivery unit cannot transition to any other state.
    This ensures the terminal state is respected.
    """
    # ARRANGE
    order = Order("O123", [0, 0], [2, 3], 100)
    driver = Driver("D1", [0, 0], 100, 120)
    env = simpy.Environment()
    delivery_unit = DeliveryUnit(order, driver, env.now)
    
    # First transition to COMPLETED
    delivery_unit.transition_to(DeliveryUnitState.COMPLETED, None, env)
    
    # ACT & ASSERT - Try to transition back to IN_PROGRESS
    with pytest.raises(ValueError):
        delivery_unit.transition_to(DeliveryUnitState.IN_PROGRESS, None, env)
    
    # Verify state remains COMPLETED
    assert delivery_unit.state == DeliveryUnitState.COMPLETED

# Test Group 4: Testing event generation
def test_state_change_generates_event():
    """
    Test that state transitions generate the appropriate events.
    This verifies the integration with the event system.
    """
    # ARRANGE
    order = Order("O123", [0, 0], [2, 3], 100)
    driver = Driver("D1", [0, 0], 100, 120)
    env = simpy.Environment()
    dispatcher = EventDispatcher()
    delivery_unit = DeliveryUnit(order, driver, env.now)
    
    # Track received events
    received_events = []
    def test_handler(event):
        received_events.append(event)
    
    # Register handler
    dispatcher.register(DeliveryUnitStateChangedEvent, test_handler)
    
    # ACT - Change state with dispatcher
    delivery_unit.transition_to(DeliveryUnitState.COMPLETED, dispatcher, env)
    
    # ASSERT - Verify event was dispatched with correct data
    assert len(received_events) == 1, "Should receive exactly one event"
    event = received_events[0]
    assert isinstance(event, DeliveryUnitStateChangedEvent)
    assert event.delivery_unit_id == delivery_unit.unit_id
    assert event.old_state == DeliveryUnitState.IN_PROGRESS
    assert event.new_state == DeliveryUnitState.COMPLETED
    assert event.timestamp == env.now

# Test Group 5: Testing assignment score recording (updated for priority scoring)
def test_assignment_score_recording():
    """
    Test that assignment scores can be properly recorded in the delivery unit.
    This is important for tracking the decision-making process with priority scoring.
    """
    # ARRANGE
    order = Order("O123", [0, 0], [2, 3], 100)
    driver = Driver("D1", [0, 0], 100, 120)
    delivery_unit = DeliveryUnit(order, driver, 110)
    
    # ACT - Set assignment scores (typically done by AssignmentService)
    delivery_unit.assignment_scores = {
        "distance_score": 0.75,        # 75% distance efficiency
        "throughput_score": 0.0,       # Single order (baseline)
        "fairness_score": 0.90,        # 90% fairness (short wait)
        "combined_score_0_1": 0.55,    # Weighted combination
        "priority_score_0_100": 55.0,  # Final priority score
        "total_distance": 8.5,         # Actual distance in km
        "num_orders": 1,               # Number of orders
        "assignment_delay_minutes": 3.0       # Wait time
    }
    
    # ASSERT - Verify scores are recorded correctly
    assert delivery_unit.assignment_scores["distance_score"] == 0.75
    assert delivery_unit.assignment_scores["throughput_score"] == 0.0
    assert delivery_unit.assignment_scores["fairness_score"] == 0.90
    assert delivery_unit.assignment_scores["combined_score_0_1"] == 0.55
    assert delivery_unit.assignment_scores["priority_score_0_100"] == 55.0
    assert delivery_unit.assignment_scores["total_distance"] == 8.5
    assert delivery_unit.assignment_scores["num_orders"] == 1
    assert delivery_unit.assignment_scores["assignment_delay_minutes"] == 3.0

# Test Group 6: Testing assignment path recording
def test_assignment_path_recording():
    """
    Test that the assignment path (immediate vs periodic) can be recorded.
    This helps track which assignment strategy was used.
    """
    # ARRANGE
    order = Order("O123", [0, 0], [2, 3], 100)
    driver = Driver("D1", [0, 0], 100, 120)
    delivery_unit = DeliveryUnit(order, driver, 110)
    
    # ACT - Set assignment path (typically done by AssignmentService)
    delivery_unit.assignment_path = "immediate"
    
    # ASSERT - Verify path is recorded
    assert delivery_unit.assignment_path == "immediate"
    
    # Test alternative path
    delivery_unit.assignment_path = "periodic"
    assert delivery_unit.assignment_path == "periodic"

# Test Group 7: Edge cases
def test_delivery_unit_with_minimal_data():
    """
    Test that a DeliveryUnit works correctly with minimal data.
    This verifies robustness when optional fields aren't set.
    """
    # ARRANGE
    order = Order("O123", [0, 0], [2, 3], 100)
    driver = Driver("D1", [0, 0], 100, 120)
    
    # ACT - Create delivery unit without setting optional fields
    delivery_unit = DeliveryUnit(order, driver, 110)
    
    # ASSERT - Verify it works without assignment scores or path set
    assert delivery_unit.state == DeliveryUnitState.IN_PROGRESS
    assert delivery_unit.assignment_path is None
    assert all(value is None for value in delivery_unit.assignment_scores.values())
    
    # Should still be able to transition state
    delivery_unit.transition_to(DeliveryUnitState.COMPLETED)
    assert delivery_unit.state == DeliveryUnitState.COMPLETED