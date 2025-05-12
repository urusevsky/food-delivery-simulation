import pytest
from delivery_sim.repositories.restaurant_repository import RestaurantRepository
from delivery_sim.entities.restaurant import Restaurant

# Test Group 1: Basic repository operations
def test_repository_initialization():
    """
    Test that a new restaurant repository starts empty with proper initialization.
    This ensures our repository has a clean starting state for storing restaurants.
    """
    # ARRANGE & ACT
    repository = RestaurantRepository()
    
    # ASSERT
    assert hasattr(repository, 'restaurants'), "Repository should have a restaurants dictionary"
    assert isinstance(repository.restaurants, dict), "Restaurants storage should be a dictionary"
    assert len(repository.restaurants) == 0, "New repository should start empty"
    assert repository.count() == 0, "Count should return 0 for empty repository"

def test_add_restaurant():
    """
    Test that restaurants can be added to the repository.
    This is fundamental for setting up the simulation's static infrastructure.
    """
    # ARRANGE
    repository = RestaurantRepository()
    restaurant = Restaurant("R1", [0, 0])
    
    # ACT
    repository.add(restaurant)
    
    # ASSERT
    assert repository.count() == 1
    assert "R1" in repository.restaurants
    assert repository.restaurants["R1"] is restaurant

def test_add_multiple_restaurants():
    """
    Test that multiple restaurants can be added and stored correctly.
    This simulates a realistic delivery area with many restaurants.
    """
    # ARRANGE
    repository = RestaurantRepository()
    restaurant1 = Restaurant("R1", [0, 0])
    restaurant2 = Restaurant("R2", [5, 5])
    restaurant3 = Restaurant("R3", [10, 10])
    
    # ACT
    repository.add(restaurant1)
    repository.add(restaurant2)
    repository.add(restaurant3)
    
    # ASSERT
    assert repository.count() == 3
    assert all(r_id in repository.restaurants for r_id in ["R1", "R2", "R3"])

# Test Group 2: Finding restaurants
def test_find_by_id_existing_restaurant():
    """
    Test that an existing restaurant can be found by its ID.
    This is crucial for order creation and restaurant lookups.
    """
    # ARRANGE
    repository = RestaurantRepository()
    restaurant = Restaurant("R1", [0, 0])
    repository.add(restaurant)
    
    # ACT
    found_restaurant = repository.find_by_id("R1")
    
    # ASSERT
    assert found_restaurant is restaurant
    assert found_restaurant.restaurant_id == "R1"

def test_find_by_id_nonexistent_restaurant():
    """
    Test that find_by_id returns None for nonexistent restaurants.
    This handles the case where a restaurant ID doesn't exist.
    """
    # ARRANGE
    repository = RestaurantRepository()
    restaurant = Restaurant("R1", [0, 0])
    repository.add(restaurant)
    
    # ACT
    found_restaurant = repository.find_by_id("R999")  # Nonexistent ID
    
    # ASSERT
    assert found_restaurant is None

def test_find_all():
    """
    Test that find_all returns all restaurants in the repository.
    This is useful for initialization and statistics about the delivery area.
    """
    # ARRANGE
    repository = RestaurantRepository()
    restaurant1 = Restaurant("R1", [0, 0])
    restaurant2 = Restaurant("R2", [5, 5])
    
    repository.add(restaurant1)
    repository.add(restaurant2)
    
    # ACT
    all_restaurants = repository.find_all()
    
    # ASSERT
    assert len(all_restaurants) == 2
    assert restaurant1 in all_restaurants
    assert restaurant2 in all_restaurants
    assert isinstance(all_restaurants, list)

# Test Group 3: Specialized queries
def test_find_by_location_exact_match():
    """
    Test finding a restaurant at an exact location.
    This is useful for determining if orders from the same location
    are from the same restaurant.
    """
    # ARRANGE
    repository = RestaurantRepository()
    restaurant1 = Restaurant("R1", [0, 0])
    restaurant2 = Restaurant("R2", [5.5, 5.5])
    repository.add(restaurant1)
    repository.add(restaurant2)
    
    # ACT
    found_restaurant = repository.find_by_location([0, 0])
    
    # ASSERT
    assert found_restaurant is restaurant1
    assert found_restaurant.location == [0, 0]

def test_find_by_location_with_tolerance():
    """
    Test finding a restaurant at a location with floating-point tolerance.
    This handles cases where coordinates might have slight variations
    due to floating-point arithmetic.
    """
    # ARRANGE
    repository = RestaurantRepository()
    restaurant = Restaurant("R1", [5.0, 5.0])
    repository.add(restaurant)
    
    # ACT - Search with slightly different coordinates
    found_restaurant = repository.find_by_location([5.0000001, 4.9999999])
    
    # ASSERT
    assert found_restaurant is restaurant
    assert found_restaurant.restaurant_id == "R1"

def test_find_by_location_custom_tolerance():
    """
    Test finding a restaurant with a custom tolerance value.
    This allows for more flexible location matching when needed.
    """
    # ARRANGE
    repository = RestaurantRepository()
    restaurant = Restaurant("R1", [5.0, 5.0])
    repository.add(restaurant)
    
    # ACT - Search with larger difference but custom tolerance
    found_restaurant = repository.find_by_location([5.1, 4.9], tolerance=0.2)
    
    # ASSERT
    assert found_restaurant is restaurant

def test_find_by_location_no_match():
    """
    Test that find_by_location returns None when no restaurant is at the location.
    This handles cases where the search location doesn't match any restaurant.
    """
    # ARRANGE
    repository = RestaurantRepository()
    restaurant1 = Restaurant("R1", [0, 0])
    restaurant2 = Restaurant("R2", [10, 10])
    repository.add(restaurant1)
    repository.add(restaurant2)
    
    # ACT
    found_restaurant = repository.find_by_location([5, 5])
    
    # ASSERT
    assert found_restaurant is None

def test_find_by_location_multiple_restaurants_returns_first():
    """
    Test that when multiple restaurants could match (within tolerance),
    the method returns the first one found.
    This documents the expected behavior in edge cases.
    """
    # ARRANGE
    repository = RestaurantRepository()
    restaurant1 = Restaurant("R1", [5.0, 5.0])
    restaurant2 = Restaurant("R2", [5.00001, 5.00001])
    repository.add(restaurant1)
    repository.add(restaurant2)
    
    # ACT - Both restaurants are within default tolerance
    found_restaurant = repository.find_by_location([5.0, 5.0])
    
    # ASSERT - Should return the first match
    assert found_restaurant is restaurant1

# Test Group 4: Edge cases
def test_duplicate_restaurant_ids():
    """
    Test what happens when trying to add a restaurant with duplicate ID.
    This verifies that the repository handles ID conflicts.
    """
    # ARRANGE
    repository = RestaurantRepository()
    restaurant1 = Restaurant("R1", [0, 0])
    restaurant2 = Restaurant("R1", [10, 10])  # Same ID!
    
    # ACT
    repository.add(restaurant1)
    repository.add(restaurant2)  # This will overwrite restaurant1
    
    # ASSERT
    assert repository.count() == 1
    found_restaurant = repository.find_by_id("R1")
    assert found_restaurant is restaurant2
    assert found_restaurant.location == [10, 10]

def test_repository_isolation():
    """
    Test that find_all returns a new list, not the internal storage.
    This ensures external code can't accidentally modify the repository.
    """
    # ARRANGE
    repository = RestaurantRepository()
    restaurant = Restaurant("R1", [0, 0])
    repository.add(restaurant)
    
    # ACT
    all_restaurants = repository.find_all()
    all_restaurants.clear()  # Try to clear the returned list
    
    # ASSERT
    assert repository.count() == 1
    assert len(repository.find_all()) == 1

def test_count_accuracy():
    """
    Test that count() accurately reflects the repository size.
    This is important for initialization verification and statistics.
    """
    # ARRANGE
    repository = RestaurantRepository()
    
    # ACT & ASSERT - Test at different sizes
    assert repository.count() == 0
    
    repository.add(Restaurant("R1", [0, 0]))
    assert repository.count() == 1
    
    repository.add(Restaurant("R2", [5, 5]))
    repository.add(Restaurant("R3", [10, 10]))
    assert repository.count() == 3
    
    # Overwrite a restaurant (same ID)
    repository.add(Restaurant("R3", [15, 15]))
    assert repository.count() == 3