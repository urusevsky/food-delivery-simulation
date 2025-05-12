# tests/unit/repositories/test_driver_repository.py
import pytest
from delivery_sim.repositories.driver_repository import DriverRepository
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.states import DriverState

# Test Group 1: Basic repository operations
def test_repository_initialization():
    """
    Test that a new driver repository starts empty with proper initialization.
    This ensures our repository has a clean starting state for storing drivers.
    """
    # ARRANGE & ACT
    repository = DriverRepository()
    
    # ASSERT
    assert hasattr(repository, 'drivers'), "Repository should have a drivers dictionary"
    assert isinstance(repository.drivers, dict), "Drivers storage should be a dictionary"
    assert len(repository.drivers) == 0, "New repository should start empty"
    assert repository.count() == 0, "Count should return 0 for empty repository"

def test_add_driver():
    """
    Test that drivers can be added to the repository.
    This verifies the fundamental storage operation works correctly.
    """
    # ARRANGE
    repository = DriverRepository()
    driver = Driver("D1", [0, 0], 100, 120)
    
    # ACT
    repository.add(driver)
    
    # ASSERT
    assert repository.count() == 1, "Repository should contain one driver"
    assert "D1" in repository.drivers, "Driver ID should be in the repository"
    assert repository.drivers["D1"] is driver, "Repository should store the exact driver object"

def test_add_multiple_drivers():
    """
    Test that multiple drivers can be added and stored correctly.
    This verifies the repository handles multiple entities properly.
    """
    # ARRANGE
    repository = DriverRepository()
    driver1 = Driver("D1", [0, 0], 100, 120)
    driver2 = Driver("D2", [1, 1], 100, 120)
    driver3 = Driver("D3", [2, 2], 100, 120)
    
    # ACT
    repository.add(driver1)
    repository.add(driver2)
    repository.add(driver3)
    
    # ASSERT
    assert repository.count() == 3, "Repository should contain three drivers"
    assert all(driver_id in repository.drivers for driver_id in ["D1", "D2", "D3"])

# Test Group 2: Finding drivers
def test_find_by_id_existing_driver():
    """
    Test that an existing driver can be found by their ID.
    This is crucial for retrieving specific drivers during simulation.
    """
    # ARRANGE
    repository = DriverRepository()
    driver = Driver("D1", [0, 0], 100, 120)
    repository.add(driver)
    
    # ACT
    found_driver = repository.find_by_id("D1")
    
    # ASSERT
    assert found_driver is driver, "Should return the exact driver object"
    assert found_driver.driver_id == "D1", "Found driver should have correct ID"

def test_find_by_id_nonexistent_driver():
    """
    Test that find_by_id returns None for nonexistent drivers.
    This handles the case where a driver ID doesn't exist.
    """
    # ARRANGE
    repository = DriverRepository()
    driver = Driver("D1", [0, 0], 100, 120)
    repository.add(driver)
    
    # ACT
    found_driver = repository.find_by_id("D999")  # Nonexistent ID
    
    # ASSERT
    assert found_driver is None, "Should return None for nonexistent driver"

def test_find_all():
    """
    Test that find_all returns all drivers in the repository.
    This is useful for global operations or statistics.
    """
    # ARRANGE
    repository = DriverRepository()
    driver1 = Driver("D1", [0, 0], 100, 120)
    driver2 = Driver("D2", [1, 1], 100, 120)
    
    repository.add(driver1)
    repository.add(driver2)
    
    # ACT
    all_drivers = repository.find_all()
    
    # ASSERT
    assert len(all_drivers) == 2, "Should return all drivers"
    assert driver1 in all_drivers, "Should include first driver"
    assert driver2 in all_drivers, "Should include second driver"
    assert isinstance(all_drivers, list), "Should return a list"

# Test Group 3: Finding by state
def test_find_by_state():
    """
    Test that drivers can be filtered by their state.
    This is essential for finding drivers at specific stages of their lifecycle.
    """
    # ARRANGE
    repository = DriverRepository()
    
    # Create drivers in different states
    driver1 = Driver("D1", [0, 0], 100, 120)  # AVAILABLE by default
    driver2 = Driver("D2", [1, 1], 100, 120)
    driver3 = Driver("D3", [2, 2], 100, 120)
    
    # Transition drivers to different states
    driver2.state = DriverState.DELIVERING
    driver3.state = DriverState.OFFLINE
    
    repository.add(driver1)
    repository.add(driver2)
    repository.add(driver3)
    
    # ACT
    available_drivers = repository.find_by_state(DriverState.AVAILABLE)
    delivering_drivers = repository.find_by_state(DriverState.DELIVERING)
    offline_drivers = repository.find_by_state(DriverState.OFFLINE)
    
    # ASSERT
    assert len(available_drivers) == 1, "Should find one AVAILABLE driver"
    assert driver1 in available_drivers, "Should find the correct AVAILABLE driver"
    
    assert len(delivering_drivers) == 1, "Should find one DELIVERING driver"
    assert driver2 in delivering_drivers, "Should find the correct DELIVERING driver"
    
    assert len(offline_drivers) == 1, "Should find one OFFLINE driver"
    assert driver3 in offline_drivers, "Should find the correct OFFLINE driver"

# Test Group 4: Specialized queries
def test_find_available_drivers():
    """
    Test the specialized method for finding available drivers.
    This is a critical operation for assignment services to find drivers 
    who can accept new deliveries.
    """
    # ARRANGE
    repository = DriverRepository()
    
    # Create drivers in various states
    available_driver1 = Driver("D1", [0, 0], 100, 120)  # AVAILABLE
    available_driver2 = Driver("D2", [1, 1], 100, 120)  # AVAILABLE
    delivering_driver = Driver("D3", [2, 2], 100, 120)
    offline_driver = Driver("D4", [3, 3], 100, 120)
    
    # Set states
    delivering_driver.state = DriverState.DELIVERING
    offline_driver.state = DriverState.OFFLINE
    
    repository.add(available_driver1)
    repository.add(available_driver2)
    repository.add(delivering_driver)
    repository.add(offline_driver)
    
    # ACT
    available_drivers = repository.find_available_drivers()
    
    # ASSERT
    assert len(available_drivers) == 2, "Should find two available drivers"
    assert available_driver1 in available_drivers
    assert available_driver2 in available_drivers
    assert delivering_driver not in available_drivers
    assert offline_driver not in available_drivers

def test_find_active_drivers():
    """
    Test finding all drivers who haven't logged out.
    This includes drivers who are AVAILABLE or DELIVERING, but not OFFLINE.
    This is useful for system monitoring and statistics.
    """
    # ARRANGE
    repository = DriverRepository()
    
    # Create drivers in various states
    available_driver = Driver("D1", [0, 0], 100, 120)  # AVAILABLE
    delivering_driver = Driver("D2", [1, 1], 100, 120)
    offline_driver1 = Driver("D3", [2, 2], 100, 120)
    offline_driver2 = Driver("D4", [3, 3], 100, 120)
    
    # Set states
    delivering_driver.state = DriverState.DELIVERING
    offline_driver1.state = DriverState.OFFLINE
    offline_driver2.state = DriverState.OFFLINE
    
    repository.add(available_driver)
    repository.add(delivering_driver)
    repository.add(offline_driver1)
    repository.add(offline_driver2)
    
    # ACT
    active_drivers = repository.find_active_drivers()
    
    # ASSERT
    assert len(active_drivers) == 2, "Should find two active drivers"
    assert available_driver in active_drivers
    assert delivering_driver in active_drivers
    assert offline_driver1 not in active_drivers
    assert offline_driver2 not in active_drivers

# Test Group 5: Edge cases
def test_duplicate_driver_ids():
    """
    Test what happens when trying to add a driver with duplicate ID.
    This verifies that the repository handles ID conflicts.
    """
    # ARRANGE
    repository = DriverRepository()
    driver1 = Driver("D1", [0, 0], 100, 120)
    driver2 = Driver("D1", [2, 2], 110, 150)  # Same ID!
    
    # ACT
    repository.add(driver1)
    repository.add(driver2)  # This will overwrite driver1
    
    # ASSERT
    assert repository.count() == 1, "Should still have only one driver"
    found_driver = repository.find_by_id("D1")
    assert found_driver is driver2, "Later driver should overwrite earlier one"
    assert found_driver.location == [2, 2], "Should have driver2's data"

def test_repository_isolation():
    """
    Test that find_all returns a new list, not the internal storage.
    This ensures external code can't accidentally modify the repository.
    """
    # ARRANGE
    repository = DriverRepository()
    driver = Driver("D1", [0, 0], 100, 120)
    repository.add(driver)
    
    # ACT
    all_drivers = repository.find_all()
    all_drivers.clear()  # Try to clear the returned list
    
    # ASSERT
    assert repository.count() == 1, "Repository should still contain the driver"
    assert len(repository.find_all()) == 1, "Driver should still be in repository"

def test_count_accuracy():
    """
    Test that count() accurately reflects the repository size.
    This is important for monitoring and statistics.
    """
    # ARRANGE
    repository = DriverRepository()
    
    # ACT & ASSERT - Test at different sizes
    assert repository.count() == 0, "Empty repository should have count 0"
    
    repository.add(Driver("D1", [0, 0], 100, 120))
    assert repository.count() == 1, "Should count 1 after adding one driver"
    
    repository.add(Driver("D2", [1, 1], 100, 120))
    repository.add(Driver("D3", [2, 2], 100, 120))
    assert repository.count() == 3, "Should count 3 after adding three drivers"
    
    # Overwrite a driver (same ID)
    repository.add(Driver("D3", [5, 5], 100, 120))
    assert repository.count() == 3, "Count shouldn't change when overwriting"