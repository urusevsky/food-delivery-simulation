import pytest
import math
from delivery_sim.utils.location_utils import calculate_distance, locations_are_equal

# Test Group 1: Testing calculate_distance function
def test_calculate_distance_horizontal():
    """Test distance calculation between two points on the same vertical line."""
    # ARRANGE
    loc1 = [0, 0]
    loc2 = [5, 0]
    
    # ACT
    distance = calculate_distance(loc1, loc2)
    
    # ASSERT
    assert distance == 5.0, "Horizontal distance should be exactly 5.0"

def test_calculate_distance_vertical():
    """Test distance calculation between two points on the same horizontal line."""
    # ARRANGE
    loc1 = [0, 0]
    loc2 = [0, 7]
    
    # ACT
    distance = calculate_distance(loc1, loc2)
    
    # ASSERT
    assert distance == 7.0, "Vertical distance should be exactly 7.0"

def test_calculate_distance_diagonal():
    """Test distance calculation between two points forming a diagonal."""
    # ARRANGE
    loc1 = [0, 0]
    loc2 = [3, 4]
    
    # ACT
    distance = calculate_distance(loc1, loc2)
    
    # ASSERT
    assert distance == 5.0, "Diagonal distance should be exactly 5.0 (3-4-5 triangle)"

def test_calculate_distance_floating_point():
    """Test distance calculation with floating point coordinates."""
    # ARRANGE
    loc1 = [1.5, 2.5]
    loc2 = [4.5, 6.5]
    
    # ACT
    distance = calculate_distance(loc1, loc2)
    
    # ASSERT
    expected = math.sqrt((4.5 - 1.5)**2 + (6.5 - 2.5)**2)
    assert distance == expected, f"Distance should be {expected}"

def test_calculate_distance_negative_coordinates():
    """Test distance calculation with negative coordinates."""
    # ARRANGE
    loc1 = [-3, -4]
    loc2 = [0, 0]
    
    # ACT
    distance = calculate_distance(loc1, loc2)
    
    # ASSERT
    assert distance == 5.0, "Distance should be exactly 5.0"

def test_calculate_distance_same_point():
    """Test distance calculation between the same point."""
    # ARRANGE
    loc = [10, 10]
    
    # ACT
    distance = calculate_distance(loc, loc)
    
    # ASSERT
    assert distance == 0.0, "Distance between same point should be 0.0"

# Test Group 2: Testing locations_are_equal function
def test_locations_are_equal_same_location():
    """Test equality check for the same location."""
    # ARRANGE
    loc = [5, 5]
    
    # ACT
    result = locations_are_equal(loc, loc)
    
    # ASSERT
    assert result is True, "Same locations should be equal"

def test_locations_are_equal_different_locations():
    """Test equality check for different locations."""
    # ARRANGE
    loc1 = [5, 5]
    loc2 = [5, 6]
    
    # ACT
    result = locations_are_equal(loc1, loc2)
    
    # ASSERT
    assert result is False, "Different locations should not be equal"

def test_locations_are_equal_different_x():
    """Test equality check for locations with same y but different x."""
    # ARRANGE
    loc1 = [5, 5]
    loc2 = [6, 5]
    
    # ACT
    result = locations_are_equal(loc1, loc2)
    
    # ASSERT
    assert result is False, "Locations with different x should not be equal"

def test_locations_are_equal_different_y():
    """Test equality check for locations with same x but different y."""
    # ARRANGE
    loc1 = [5, 5]
    loc2 = [5, 6]
    
    # ACT
    result = locations_are_equal(loc1, loc2)
    
    # ASSERT
    assert result is False, "Locations with different y should not be equal"

def test_locations_are_equal_floating_point():
    """Test equality check with floating point coordinates."""
    # ARRANGE
    loc1 = [5.0, 7.0]
    loc2 = [5.0, 7.0]
    
    # ACT
    result = locations_are_equal(loc1, loc2)
    
    # ASSERT
    assert result is True, "Same floating point locations should be equal"

def test_locations_are_equal_negative_coordinates():
    """Test equality check with negative coordinates."""
    # ARRANGE
    loc1 = [-3, -4]
    loc2 = [-3, -4]
    
    # ACT
    result = locations_are_equal(loc1, loc2)
    
    # ASSERT
    assert result is True, "Same negative coordinates should be equal"