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


