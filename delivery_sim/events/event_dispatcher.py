class EventDispatcher:
    """
    Central hub for event broadcasting in the system.
    
    The EventDispatcher maintains a registry of event handlers and
    delivers events to all handlers that have registered for them.
    """
    
    def __init__(self):
        """Initialize an empty handler registry."""
        self.handlers = {}  # Maps event types to lists of handler functions
    
    def register(self, event_type, handler):
        """
        Register a handler function for a specific event type.
        
        Args:
            event_type: The class of event to handle (not an instance)
            handler: Function to call when events of this type are dispatched
        """
        # First condition: Initialize the container for this event type if needed
        if event_type not in self.handlers:
            self.handlers[event_type] = []
            
        # Second condition: Only add the handler if it's not already registered
        if handler not in self.handlers[event_type]:
            self.handlers[event_type].append(handler)
            
    def unregister(self, event_type, handler):
        """
        Remove a handler function for a specific event type.
        
        Args:
            event_type: The class of event to handle (not an instance)
            handler: Function to remove from handlers
        """
        if event_type in self.handlers and handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)
            
            # Clean up empty handler lists
            if not self.handlers[event_type]:
                del self.handlers[event_type]
    
    def dispatch(self, event):
        """
        Send an event to all registered handlers.
        
        Args:
            event: The event instance to dispatch
        """
        event_type = type(event)
        
        # Direct handlers for this specific event type
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                handler(event)
                
        # We could also implement parent class handling if needed
        # This would allow handlers to register for parent event types
        # and receive all events of child types