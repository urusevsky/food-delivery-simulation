class RestaurantRepository:
    """
    Repository for managing restaurants in the simulation.
    
    This repository stores all restaurants created during simulation initialization
    and provides methods for finding restaurants based on various criteria.
    """
    
    def __init__(self):
        """Initialize an empty restaurant repository."""
        self.restaurants = {}  # Maps restaurant_id to Restaurant objects
    
    def add(self, restaurant):
        """
        Add a restaurant to the repository.
        
        Args:
            restaurant: The Restaurant object to add
        """
        self.restaurants[restaurant.restaurant_id] = restaurant
    
    def find_by_id(self, restaurant_id):
        """
        Find a restaurant by its ID.
        
        Args:
            restaurant_id: The ID of the restaurant to find
            
        Returns:
            Restaurant: The found restaurant or None if not found
        """
        return self.restaurants.get(restaurant_id)
    
    def find_all(self):
        """
        Get all restaurants in the repository.
        
        Returns:
            list: All Restaurant objects in the repository
        """
        return list(self.restaurants.values())
    
    def find_by_location(self, location, tolerance=1e-6):
        """
        Find a restaurant at a specific location.
        
        This is useful when working with coordinates that may have slight
        floating-point differences but represent the same location.
        
        Args:
            location: [x, y] coordinates to search for
            tolerance: Maximum distance for location match (default: 1e-6)
            
        Returns:
            Restaurant: First restaurant found at the location, or None
        """
        for restaurant in self.restaurants.values():
            # Check if coordinates are within tolerance
            x_diff = abs(restaurant.location[0] - location[0])
            y_diff = abs(restaurant.location[1] - location[1])
            
            if x_diff <= tolerance and y_diff <= tolerance:
                return restaurant
        
        return None
    
    def count(self):
        """
        Get the total number of restaurants in the repository.
        
        Returns:
            int: The number of restaurants
        """
        return len(self.restaurants)