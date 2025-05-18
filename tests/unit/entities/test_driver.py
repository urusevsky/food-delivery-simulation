# tests/unit/entities/test_driver.py
import pytest
import simpy
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.states import DriverState
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.events.driver_events import DriverStateChangedEvent
from delivery_sim.utils.entity_type_utils import EntityType

def test_driver_initialization():
    """Test that a Driver is properly initialized with default values."""
    driver_id = "D1"
    initial_location = [0, 0]
    login_time = 100
    service_duration = 120
    
    driver = Driver(driver_id, initial_location, login_time, service_duration)
    
    assert driver.driver_id == driver_id
    assert driver.location == initial_location
    assert driver.login_time == login_time
    assert driver.service_duration == service_duration
    assert driver.intended_logout_time == login_time + service_duration
    assert driver.state == DriverState.AVAILABLE
    assert driver.current_delivery_unit is None
    assert driver.last_state_change_time == login_time
    assert driver.entity_type == EntityType.DRIVER

def test_valid_state_transitions():
    """Test that valid state transitions work properly."""
    # Test main "happy path" flow
    driver1 = Driver("D1", [0, 0], 100, 120)
    
    driver1.transition_to(DriverState.DELIVERING)
    assert driver1.state == DriverState.DELIVERING
    
    driver1.transition_to(DriverState.AVAILABLE)
    assert driver1.state == DriverState.AVAILABLE
    
    driver1.transition_to(DriverState.OFFLINE)
    assert driver1.state == DriverState.OFFLINE
    
    # Test direct path from AVAILABLE to OFFLINE (for drivers at logout time with no tasks)
    driver3 = Driver("D3", [0, 0], 100, 120)
    assert driver3.state == DriverState.AVAILABLE
    
    # Driver's logout time arrives while available
    driver3.transition_to(DriverState.OFFLINE)
    assert driver3.state == DriverState.OFFLINE

def test_invalid_state_transitions():
    """Test that invalid state transitions raise appropriate errors."""
    driver = Driver("D1", [0, 0], 100, 120)
    
    # Test DELIVERING -> OFFLINE (should fail with updated valid transitions)
    driver.transition_to(DriverState.DELIVERING)
    with pytest.raises(ValueError):
        driver.transition_to(DriverState.OFFLINE)
    
    # Test OFFLINE -> any state (should fail)
    driver = Driver("D2", [0, 0], 100, 120)
    driver.transition_to(DriverState.OFFLINE)
    
    with pytest.raises(ValueError):
        driver.transition_to(DriverState.AVAILABLE)
    
    with pytest.raises(ValueError):
        driver.transition_to(DriverState.DELIVERING)

def test_location_update():
    """Test that driver location updates work properly."""
    driver = Driver("D1", [0, 0], 100, 120)
    
    new_location = [3, 4]
    driver.update_location(new_location)
    
    assert driver.location == new_location

def test_can_logout():
    """Test the logout capability determination."""
    driver = Driver("D1", [0, 0], 100, 120)
    
    # Driver should be able to log out when AVAILABLE
    assert driver.can_logout() is True
    
    # Driver should not be able to log out when DELIVERING
    driver.transition_to(DriverState.DELIVERING)
    assert driver.can_logout() is False
    
    # Driver should not be able to log out when OFFLINE (already logged out)
    driver = Driver("D2", [0, 0], 100, 120)
    driver.transition_to(DriverState.OFFLINE)
    assert driver.can_logout() is False

def test_driver_state_change_dispatches_event():
    """Test that state changes generate events when a dispatcher is provided."""
    # Setup
    env = simpy.Environment()
    dispatcher = EventDispatcher()
    driver = Driver("D1", [0, 0], 100, 120)
    
    # Track received events
    received_events = []
    def test_handler(event):
        received_events.append(event)
    
    # Register handler
    dispatcher.register(DriverStateChangedEvent, test_handler)
    
    # Change state with dispatcher and env
    driver.transition_to(DriverState.DELIVERING, dispatcher, env)
    
    # Verify event was dispatched with correct data
    assert len(received_events) == 1
    event = received_events[0]
    assert isinstance(event, DriverStateChangedEvent)
    assert event.driver_id == "D1"
    assert event.old_state == DriverState.AVAILABLE
    assert event.new_state == DriverState.DELIVERING
    assert event.timestamp == env.now