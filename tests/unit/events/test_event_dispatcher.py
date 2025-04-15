import pytest
from delivery_sim.events.base_events import Event
from delivery_sim.events.event_dispatcher import EventDispatcher

class TestEvent(Event):
    """Simple event class for testing."""
    def __init__(self, timestamp, data):
        super().__init__(timestamp)
        self.data = data

def test_dispatcher_registration():
    """Test that handlers can be registered and unregistered."""
    dispatcher = EventDispatcher()
    
    # Define a simple handler function
    def handler(event):
        pass
    
    # Register the handler
    dispatcher.register(TestEvent, handler)
    
    # Verify it was registered
    assert TestEvent in dispatcher.handlers
    assert handler in dispatcher.handlers[TestEvent]
    
    # Unregister the handler
    dispatcher.unregister(TestEvent, handler)
    
    # Verify it was unregistered
    assert TestEvent not in dispatcher.handlers

def test_event_dispatch():
    """Test that events are properly dispatched to handlers."""
    dispatcher = EventDispatcher()
    received_events = []
    
    # Define handler that records received events
    def handler(event):
        received_events.append(event)
    
    # Register the handler
    dispatcher.register(TestEvent, handler)
    
    # Create and dispatch an event
    test_event = TestEvent(100, "test data")
    dispatcher.dispatch(test_event)
    
    # Verify handler received the event
    assert len(received_events) == 1
    assert received_events[0] is test_event
    assert received_events[0].timestamp == 100
    assert received_events[0].data == "test data"