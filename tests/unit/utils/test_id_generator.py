# tests/unit/utils/test_id_generator.py
import pytest
from delivery_sim.utils.id_generator import PrefixedIdGenerator

# Test Group 1: Basic initialization and functionality
def test_prefixed_id_generator_initialization():
    """Test that PrefixedIdGenerator initializes with correct default values."""
    # ARRANGE & ACT
    generator = PrefixedIdGenerator('O')
    
    # ASSERT
    assert generator.prefix == 'O', "Prefix should be set to 'O'"
    assert generator.counter == 1, "Counter should start at 1 by default"
    assert generator.separator == '', "Separator should be empty string by default"

def test_next_generates_sequential_ids():
    """Test that next() method generates sequential IDs with the prefix."""
    # ARRANGE
    generator = PrefixedIdGenerator('O')
    
    # ACT
    id1 = generator.next()
    id2 = generator.next()
    id3 = generator.next()
    
    # ASSERT
    assert id1 == 'O1', "First ID should be 'O1'"
    assert id2 == 'O2', "Second ID should be 'O2'"
    assert id3 == 'O3', "Third ID should be 'O3'"
    assert generator.counter == 4, "Counter should be incremented to 4 after generating 3 IDs"

# Test Group 2: Standard entity prefix formats
def test_order_id_format():
    """Test that order ID generator produces correct format: O#"""
    # ARRANGE
    order_generator = PrefixedIdGenerator('O')
    
    # ACT
    order_id = order_generator.next()
    
    # ASSERT
    assert order_id == 'O1', "Order IDs should follow format O#"

def test_driver_id_format():
    """Test that driver ID generator produces correct format: D#"""
    # ARRANGE
    driver_generator = PrefixedIdGenerator('D')
    
    # ACT
    driver_id = driver_generator.next()
    
    # ASSERT
    assert driver_id == 'D1', "Driver IDs should follow format D#"

def test_restaurant_id_format():
    """Test that restaurant ID generator produces correct format: R#"""
    # ARRANGE
    restaurant_generator = PrefixedIdGenerator('R')
    
    # ACT
    restaurant_id = restaurant_generator.next()
    
    # ASSERT
    assert restaurant_id == 'R1', "Restaurant IDs should follow format R#"

# Test Group 3: Multiple independent generators
def test_multiple_generators():
    """Test that multiple generators are independent of each other."""
    # ARRANGE
    order_generator = PrefixedIdGenerator('O')
    driver_generator = PrefixedIdGenerator('D')
    restaurant_generator = PrefixedIdGenerator('R')
    
    # ACT - Generate IDs from each generator
    order_id1 = order_generator.next()
    driver_id1 = driver_generator.next()
    restaurant_id1 = restaurant_generator.next()
    order_id2 = order_generator.next()
    
    # ASSERT
    assert order_id1 == 'O1' and order_id2 == 'O2', "Order IDs should sequence independently"
    assert driver_id1 == 'D1', "Driver IDs should sequence independently"
    assert restaurant_id1 == 'R1', "Restaurant IDs should sequence independently"

# Test Group 4: Custom configuration options
def test_custom_starting_counter():
    """Test generator with a custom starting counter value."""
    # ARRANGE
    generator = PrefixedIdGenerator('O', start=100)
    
    # ACT
    id1 = generator.next()
    id2 = generator.next()
    
    # ASSERT
    assert id1 == 'O100', "First ID should start at the specified value"
    assert id2 == 'O101', "IDs should increment from the starting value"

def test_with_separator():
    """Test generator with a custom separator between prefix and counter."""
    # ARRANGE
    generator = PrefixedIdGenerator('O', separator='-')
    
    # ACT
    id1 = generator.next()
    id2 = generator.next()
    
    # ASSERT
    assert id1 == 'O-1', "ID should contain the separator"
    assert id2 == 'O-2', "ID should contain the separator"