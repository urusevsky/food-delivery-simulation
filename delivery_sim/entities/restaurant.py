class Restaurant:
    """
    Represents a restaurant in the delivery system.
    
    This is a simple entity with a fixed location that serves as the 
    source of orders. Unlike other entities, restaurants don't have
    state changes or complex behavior - they're primarily static 
    infrastructure elements.
    """
    
    def __init__(self, restaurant_id, location):
        """
        Initialize a new restaurant.
        
        Args:
            restaurant_id: Unique identifier for this restaurant
            location: [x, y] coordinates of the restaurant
        """
        self.restaurant_id = restaurant_id
        self.location = location
    
    def __str__(self):
        """String representation of the restaurant"""
        return f"Restaurant(id={self.restaurant_id}, location={self.location})"