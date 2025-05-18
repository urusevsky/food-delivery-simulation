# tests/unit/utils/test_entity_type_utils.py
import pytest
from delivery_sim.utils.entity_type_utils import EntityType

# Test Group 1: EntityType constants
def test_entity_type_constants():
    """Test that EntityType contains all expected type constants."""
    # ASSERT
    assert EntityType.ORDER == "order", "ORDER constant should be 'order'"
    assert EntityType.DRIVER == "driver", "DRIVER constant should be 'driver'"
    assert EntityType.RESTAURANT == "restaurant", "RESTAURANT constant should be 'restaurant'"
    assert EntityType.PAIR == "pair", "PAIR constant should be 'pair'"
    assert EntityType.DELIVERY_UNIT == "delivery_unit", "DELIVERY_UNIT constant should be 'delivery_unit'"
    
    # Verify ALL_TYPES contains all types
    assert EntityType.ALL_TYPES == {
        "order", "driver", "restaurant", "pair", "delivery_unit"
    }, "ALL_TYPES should contain all defined entity types"

# Test Group 2: is_valid method
def test_is_valid_with_valid_types():
    """Test that is_valid returns True for all valid entity types."""
    # ACT & ASSERT
    assert EntityType.is_valid("order") is True, "ORDER should be valid"
    assert EntityType.is_valid("driver") is True, "DRIVER should be valid"
    assert EntityType.is_valid("restaurant") is True, "RESTAURANT should be valid"
    assert EntityType.is_valid("pair") is True, "PAIR should be valid"
    assert EntityType.is_valid("delivery_unit") is True, "DELIVERY_UNIT should be valid"

def test_is_valid_with_invalid_types():
    """Test that is_valid returns False for invalid entity types."""
    # ACT & ASSERT
    assert EntityType.is_valid("customer") is False, "'customer' is not a valid entity type"
    assert EntityType.is_valid("DRIVER") is False, "Entity types are case-sensitive"
    assert EntityType.is_valid("") is False, "Empty string is not a valid entity type"
    assert EntityType.is_valid(None) is False, "None is not a valid entity type"
    assert EntityType.is_valid(123) is False, "Non-string values are not valid entity types"