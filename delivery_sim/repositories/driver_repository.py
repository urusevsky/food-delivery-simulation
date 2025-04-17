from delivery_sim.entities.states import DriverState

class DriverRepository:
    """
    Repository for managing drivers in the simulation.
    
    This repository stores all drivers created during the simulation and
    provides methods for finding drivers based on various criteria.
    """
    
    def __init__(self):
        """Initialize an empty driver repository."""
        self.drivers = {}  # Maps driver_id to Driver objects
    
    def add(self, driver):
        """
        Add a driver to the repository.
        
        Args:
            driver: The Driver object to add
        """
        self.drivers[driver.driver_id] = driver
    
    def find_by_id(self, driver_id):
        """
        Find a driver by their ID.
        
        Args:
            driver_id: The ID of the driver to find
            
        Returns:
            Driver: The found driver or None if not found
        """
        return self.drivers.get(driver_id)
    
    def find_all(self):
        """
        Get all drivers in the repository.
        
        Returns:
            list: All Driver objects in the repository
        """
        return list(self.drivers.values())
    
    def find_by_state(self, state):
        """
        Find all drivers in a specific state.
        
        Args:
            state: The DriverState to filter by
            
        Returns:
            list: Driver objects with the specified state
        """
        return [driver for driver in self.drivers.values() if driver.state == state]
    
    def find_available_drivers(self):
        """
        Find drivers available for assignment.
        
        Returns:
            list: Available Driver objects
        """
        return self.find_by_state(DriverState.AVAILABLE)
    
    def find_active_drivers(self):
        """
        Find all drivers who haven't logged out.
        
        Returns:
            list: Driver objects that are still active
        """
        return [driver for driver in self.drivers.values() 
                if driver.state != DriverState.OFFLINE]
    
    def count(self):
        """
        Get the total number of drivers in the repository.
        
        Returns:
            int: The number of drivers
        """
        return len(self.drivers)