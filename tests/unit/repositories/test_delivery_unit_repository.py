import pytest
from delivery_sim.repositories.delivery_unit_repository import DeliveryUnitRepository
from delivery_sim.entities.delivery_unit import DeliveryUnit
from delivery_sim.entities.states import DeliveryUnitState
from delivery_sim.entities.order import Order
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.pair import Pair

# Test Group 1: Basic repository operations
def test_repository_initialization():
    """
    Test that a new delivery unit repository starts empty with proper initialization.
    This ensures our repository has a clean starting state for storing active deliveries.
    """
    # ARRANGE & ACT
    repository = DeliveryUnitRepository()
    
    # ASSERT
    assert hasattr(repository, 'delivery_units'), "Repository should have a delivery_units dictionary"
    assert isinstance(repository.delivery_units, dict), "Delivery units storage should be a dictionary"
    assert len(repository.delivery_units) == 0, "New repository should start empty"
    assert repository.find_all() == [], "find_all should return empty list"

def test_add_delivery_unit():
    """
    Test that delivery units can be added to the repository.
    This is fundamental for tracking active deliveries in the system.
    """
    # ARRANGE
    repository = DeliveryUnitRepository()
    order = Order("O123", [0, 0], [2, 3], 100)
    driver = Driver("D1", [0, 0], 100, 120)
    delivery_unit = DeliveryUnit(order, driver, 110)
    
    # ACT
    repository.add(delivery_unit)
    
    # ASSERT
    assert delivery_unit.unit_id in repository.delivery_units
    assert repository.delivery_units[delivery_unit.unit_id] is delivery_unit
    assert len(repository.delivery_units) == 1

def test_add_multiple_delivery_units():
    """
    Test that multiple delivery units can be added and stored correctly.
    This simulates a real system where multiple deliveries are happening concurrently.
    """
    # ARRANGE
    repository = DeliveryUnitRepository()
    
    # Create first delivery unit (single order)
    order1 = Order("O101", [0, 0], [1, 1], 100)
    driver1 = Driver("D1", [0, 0], 100, 120)
    unit1 = DeliveryUnit(order1, driver1, 110)
    
    # Create second delivery unit (pair)
    order2_1 = Order("O102", [1, 1], [2, 2], 105)
    order2_2 = Order("O103", [1, 1], [3, 3], 110)
    pair2 = Pair(order2_1, order2_2, 115)
    driver2 = Driver("D2", [1, 1], 100, 120)
    unit2 = DeliveryUnit(pair2, driver2, 120)
    
    # ACT
    repository.add(unit1)
    repository.add(unit2)
    
    # ASSERT
    assert len(repository.delivery_units) == 2
    assert all(unit_id in repository.delivery_units for unit_id in [unit1.unit_id, unit2.unit_id])

# Test Group 2: Finding delivery units
def test_find_by_id_existing_unit():
    """
    Test that an existing delivery unit can be found by its ID.
    This is crucial for tracking specific deliveries during their lifecycle.
    """
    # ARRANGE
    repository = DeliveryUnitRepository()
    order = Order("O123", [0, 0], [2, 3], 100)
    driver = Driver("D1", [0, 0], 100, 120)
    delivery_unit = DeliveryUnit(order, driver, 110)
    repository.add(delivery_unit)
    
    # ACT
    found_unit = repository.find_by_id(delivery_unit.unit_id)
    
    # ASSERT
    assert found_unit is delivery_unit
    assert found_unit.unit_id == delivery_unit.unit_id

def test_find_by_id_nonexistent_unit():
    """
    Test that find_by_id returns None for nonexistent delivery units.
    This handles the case where a unit ID doesn't exist.
    """
    # ARRANGE
    repository = DeliveryUnitRepository()
    order = Order("O123", [0, 0], [2, 3], 100)
    driver = Driver("D1", [0, 0], 100, 120)
    delivery_unit = DeliveryUnit(order, driver, 110)
    repository.add(delivery_unit)
    
    # ACT
    found_unit = repository.find_by_id("DU-O-999-D999")  # Nonexistent ID
    
    # ASSERT
    assert found_unit is None

def test_find_all():
    """
    Test that find_all returns all delivery units in the repository.
    This is useful for getting a complete picture of all active deliveries.
    """
    # ARRANGE
    repository = DeliveryUnitRepository()
    
    # Create delivery units
    order1 = Order("O101", [0, 0], [1, 1], 100)
    driver1 = Driver("D1", [0, 0], 100, 120)
    unit1 = DeliveryUnit(order1, driver1, 110)
    
    order2 = Order("O102", [1, 1], [2, 2], 105)
    driver2 = Driver("D2", [1, 1], 100, 120)
    unit2 = DeliveryUnit(order2, driver2, 115)
    
    repository.add(unit1)
    repository.add(unit2)
    
    # ACT
    all_units = repository.find_all()
    
    # ASSERT
    assert len(all_units) == 2
    assert unit1 in all_units
    assert unit2 in all_units
    assert isinstance(all_units, list)

# Test Group 3: Finding by state
def test_find_by_state():
    """
    Test that delivery units can be filtered by their state.
    This is essential for finding deliveries at specific stages of completion.
    """
    # ARRANGE
    repository = DeliveryUnitRepository()
    
    # Create delivery units in different states
    order1 = Order("O101", [0, 0], [1, 1], 100)
    driver1 = Driver("D1", [0, 0], 100, 120)
    unit1 = DeliveryUnit(order1, driver1, 110)  # IN_PROGRESS by default
    
    order2 = Order("O102", [1, 1], [2, 2], 105)
    driver2 = Driver("D2", [1, 1], 100, 120)
    unit2 = DeliveryUnit(order2, driver2, 115)
    unit2.state = DeliveryUnitState.COMPLETED
    
    repository.add(unit1)
    repository.add(unit2)
    
    # ACT
    in_progress_units = repository.find_by_state(DeliveryUnitState.IN_PROGRESS)
    completed_units = repository.find_by_state(DeliveryUnitState.COMPLETED)
    
    # ASSERT
    assert len(in_progress_units) == 1
    assert unit1 in in_progress_units
    
    assert len(completed_units) == 1
    assert unit2 in completed_units

# Test Group 4: Specialized queries
def test_find_active_deliveries():
    """
    Test finding all active (in-progress) deliveries.
    This is a convenience method that filters for IN_PROGRESS state,
    useful for monitoring current system load.
    """
    # ARRANGE
    repository = DeliveryUnitRepository()
    
    # Create mix of active and completed deliveries
    order1 = Order("O101", [0, 0], [1, 1], 100)
    driver1 = Driver("D1", [0, 0], 100, 120)
    active_unit1 = DeliveryUnit(order1, driver1, 110)  # IN_PROGRESS
    
    order2 = Order("O102", [1, 1], [2, 2], 105)
    driver2 = Driver("D2", [1, 1], 100, 120)
    active_unit2 = DeliveryUnit(order2, driver2, 115)  # IN_PROGRESS
    
    order3 = Order("O103", [2, 2], [3, 3], 110)
    driver3 = Driver("D3", [2, 2], 100, 120)
    completed_unit = DeliveryUnit(order3, driver3, 120)
    completed_unit.state = DeliveryUnitState.COMPLETED
    
    repository.add(active_unit1)
    repository.add(active_unit2)
    repository.add(completed_unit)
    
    # ACT
    active_deliveries = repository.find_active_deliveries()
    
    # ASSERT
    assert len(active_deliveries) == 2
    assert active_unit1 in active_deliveries
    assert active_unit2 in active_deliveries
    assert completed_unit not in active_deliveries

def test_find_by_driver_id():
    """
    Test finding all delivery units assigned to a specific driver.
    This is crucial for tracking what a particular driver is doing or has done.
    """
    # ARRANGE
    repository = DeliveryUnitRepository()
    
    # Create delivery units for different drivers
    order1 = Order("O101", [0, 0], [1, 1], 100)
    driver1 = Driver("D1", [0, 0], 100, 120)
    unit1 = DeliveryUnit(order1, driver1, 110)
    
    order2 = Order("O102", [1, 1], [2, 2], 105)
    unit2 = DeliveryUnit(order2, driver1, 115)  # Same driver as unit1
    
    order3 = Order("O103", [2, 2], [3, 3], 110)
    driver2 = Driver("D2", [2, 2], 100, 120)
    unit3 = DeliveryUnit(order3, driver2, 120)  # Different driver
    
    repository.add(unit1)
    repository.add(unit2)
    repository.add(unit3)
    
    # ACT
    driver1_units = repository.find_by_driver_id("D1")
    driver2_units = repository.find_by_driver_id("D2")
    driver3_units = repository.find_by_driver_id("D3")  # Non-existent driver
    
    # ASSERT
    assert len(driver1_units) == 2
    assert unit1 in driver1_units
    assert unit2 in driver1_units
    
    assert len(driver2_units) == 1
    assert unit3 in driver2_units
    
    assert len(driver3_units) == 0

def test_find_by_assignment_path():
    """
    Test finding delivery units by their assignment path (immediate vs periodic).
    This is useful for analyzing how different assignment strategies perform.
    """
    # ARRANGE
    repository = DeliveryUnitRepository()
    
    # Create delivery units with different assignment paths
    order1 = Order("O101", [0, 0], [1, 1], 100)
    driver1 = Driver("D1", [0, 0], 100, 120)
    immediate_unit1 = DeliveryUnit(order1, driver1, 110)
    immediate_unit1.assignment_path = "immediate"
    
    order2 = Order("O102", [1, 1], [2, 2], 105)
    driver2 = Driver("D2", [1, 1], 100, 120)
    immediate_unit2 = DeliveryUnit(order2, driver2, 115)
    immediate_unit2.assignment_path = "immediate"
    
    order3 = Order("O103", [2, 2], [3, 3], 110)
    driver3 = Driver("D3", [2, 2], 100, 120)
    periodic_unit = DeliveryUnit(order3, driver3, 120)
    periodic_unit.assignment_path = "periodic"
    
    order4 = Order("O104", [3, 3], [4, 4], 115)
    driver4 = Driver("D4", [3, 3], 100, 120)
    no_path_unit = DeliveryUnit(order4, driver4, 125)
    # assignment_path left as None
    
    repository.add(immediate_unit1)
    repository.add(immediate_unit2)
    repository.add(periodic_unit)
    repository.add(no_path_unit)
    
    # ACT
    immediate_units = repository.find_by_assignment_path("immediate")
    periodic_units = repository.find_by_assignment_path("periodic")
    unknown_path_units = repository.find_by_assignment_path("unknown")
    
    # ASSERT
    assert len(immediate_units) == 2
    assert immediate_unit1 in immediate_units
    assert immediate_unit2 in immediate_units
    
    assert len(periodic_units) == 1
    assert periodic_unit in periodic_units
    
    assert len(unknown_path_units) == 0

# Test Group 5: Edge cases
def test_duplicate_delivery_unit_ids():
    """
    Test what happens when trying to add a delivery unit with duplicate ID.
    While this shouldn't happen in normal operation due to how IDs are generated,
    it's good to verify the repository behavior.
    """
    # ARRANGE
    repository = DeliveryUnitRepository()
    
    # Create first unit
    order1 = Order("O101", [0, 0], [1, 1], 100)
    driver1 = Driver("D1", [0, 0], 100, 120)
    unit1 = DeliveryUnit(order1, driver1, 110)
    
    # Force the same ID on a different unit (wouldn't happen normally)
    order2 = Order("O102", [1, 1], [2, 2], 105)
    driver2 = Driver("D2", [1, 1], 100, 120)
    unit2 = DeliveryUnit(order2, driver2, 115)
    unit2.unit_id = unit1.unit_id  # Force duplicate ID
    
    # ACT
    repository.add(unit1)
    repository.add(unit2)  # This will overwrite unit1
    
    # ASSERT
    assert len(repository.delivery_units) == 1
    found_unit = repository.find_by_id(unit1.unit_id)
    assert found_unit is unit2

def test_repository_isolation():
    """
    Test that find_all returns a new list, not the internal storage.
    This ensures external code can't accidentally modify the repository.
    """
    # ARRANGE
    repository = DeliveryUnitRepository()
    order = Order("O101", [0, 0], [1, 1], 100)
    driver = Driver("D1", [0, 0], 100, 120)
    unit = DeliveryUnit(order, driver, 110)
    repository.add(unit)
    
    # ACT
    all_units = repository.find_all()
    all_units.clear()  # Try to clear the returned list
    
    # ASSERT
    assert len(repository.delivery_units) == 1
    assert len(repository.find_all()) == 1