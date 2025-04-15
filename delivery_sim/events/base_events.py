class Event:
    """
    Base class for all events in the system.
    
    Events represent things that have already happened in the system, 
    rather than commands or intentions.
    """
    
    def __init__(self, timestamp):
        """
        Initialize a new event.
        
        Args:
            timestamp: When this event occurred in simulation time
        """
        self.timestamp = timestamp
        
    def __str__(self):
        """Create a string representation of the event."""
        return f"{self.__class__.__name__}(timestamp={self.timestamp})"