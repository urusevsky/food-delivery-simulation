class EntityType:
    """Constants for entity types used throughout the simulation."""
    ORDER = "order"
    DRIVER = "driver"
    RESTAURANT = "restaurant"
    PAIR = "pair"
    DELIVERY_UNIT = "delivery_unit"
    
    # Collection of all valid types for validation
    ALL_TYPES = {ORDER, DRIVER, RESTAURANT, PAIR, DELIVERY_UNIT}
    
    @classmethod
    def is_valid(cls, entity_type):
        """Check if a given string is a valid entity type."""
        return entity_type in cls.ALL_TYPES


def get_entity_type_from_id(entity_id):
    """
    Infer entity type from ID pattern.
    
    Args:
        entity_id: The entity ID string
        
    Returns:
        str: The entity type, or None if pattern not recognized
    """
    if not entity_id or not isinstance(entity_id, str):
        return None
    
    # Check longer prefixes first to avoid false matches
    if entity_id.startswith("DU"):
        return EntityType.DELIVERY_UNIT
    elif entity_id.startswith("P"):
        return EntityType.PAIR
    elif entity_id.startswith("O"):
        return EntityType.ORDER
    elif entity_id.startswith("D"):
        return EntityType.DRIVER
    elif entity_id.startswith("R"):
        return EntityType.RESTAURANT
    
    return None

