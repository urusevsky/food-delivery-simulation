import pytest
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.states import DriverState

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
    assert driver.current_delivery is None
    assert driver.last_state_change_time == login_time

def test_valid_state_transitions():
    """Test that valid state transitions work properly."""
    # Test main "happy path" flow
    driver1 = Driver("D1", [0, 0], 100, 120)
    
    driver1.transition_to(DriverState.ASSIGNED)
    assert driver1.state == DriverState.ASSIGNED
    
    driver1.transition_to(DriverState.PICKING_UP)
    assert driver1.state == DriverState.PICKING_UP
    
    driver1.transition_to(DriverState.DELIVERING)
    assert driver1.state == DriverState.DELIVERING
    
    driver1.transition_to(DriverState.AVAILABLE)
    assert driver1.state == DriverState.AVAILABLE
    
    driver1.transition_to(DriverState.OFFLINE)
    assert driver1.state == DriverState.OFFLINE
    
    # Test direct path from DELIVERING to OFFLINE (for drivers past their logout time)
    driver2 = Driver("D2", [0, 0], 100, 120)
    
    driver2.transition_to(DriverState.ASSIGNED)
    assert driver2.state == DriverState.ASSIGNED
    
    driver2.transition_to(DriverState.PICKING_UP)
    assert driver2.state == DriverState.PICKING_UP
    
    driver2.transition_to(DriverState.DELIVERING)
    assert driver2.state == DriverState.DELIVERING
    
    # Driver completes delivery past their logout time
    driver2.transition_to(DriverState.OFFLINE)
    assert driver2.state == DriverState.OFFLINE
    
    # Test direct path from AVAILABLE to OFFLINE (for drivers at logout time with no tasks)
    driver3 = Driver("D3", [0, 0], 100, 120)
    assert driver3.state == DriverState.AVAILABLE
    
    # Driver's logout time arrives while available
    driver3.transition_to(DriverState.OFFLINE)
    assert driver3.state == DriverState.OFFLINE

def test_invalid_state_transitions():
    """Test that invalid state transitions raise appropriate errors."""
    driver = Driver("D1", [0, 0], 100, 120)
    
    # Test AVAILABLE -> DELIVERING (should fail)
    with pytest.raises(ValueError):
        driver.transition_to(DriverState.DELIVERING)
    
    # Move to ASSIGNED state
    driver.transition_to(DriverState.ASSIGNED)
    
    # Test ASSIGNED -> AVAILABLE (should fail)
    with pytest.raises(ValueError):
        driver.transition_to(DriverState.AVAILABLE)
    
    # Test ASSIGNED -> OFFLINE (should fail)
    with pytest.raises(ValueError):
        driver.transition_to(DriverState.OFFLINE)

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
    
    # Driver should not be able to log out when ASSIGNED
    driver.transition_to(DriverState.ASSIGNED)
    assert driver.can_logout() is False
    
    # Driver should not be able to log out when PICKING_UP
    driver.transition_to(DriverState.PICKING_UP)
    assert driver.can_logout() is False
    
    # Driver should not be able to log out when DELIVERING
    driver.transition_to(DriverState.DELIVERING)
    assert driver.can_logout() is False
    
    # Driver should be able to log out when back to AVAILABLE
    driver.transition_to(DriverState.AVAILABLE)
    assert driver.can_logout() is True