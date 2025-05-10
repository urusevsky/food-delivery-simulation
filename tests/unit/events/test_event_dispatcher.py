# tests/unit/events/test_event_dispatcher.py
import pytest
from delivery_sim.events.base_events import Event
from delivery_sim.events.event_dispatcher import EventDispatcher

class TestEvent(Event):
    """Simple event class for testing purposes."""
    def __init__(self, timestamp, data):
        super().__init__(timestamp)
        self.data = data

class ChildTestEvent(TestEvent):
    """Child event class for testing inheritance handling."""
    pass

class OtherEvent(Event):
    """Different event type for testing event type filtering."""
    def __init__(self, timestamp, info):
        super().__init__(timestamp)
        self.info = info

def test_dispatcher_initialization():
    """Test that a new dispatcher starts with an empty handler registry."""
    dispatcher = EventDispatcher()
    assert hasattr(dispatcher, 'handlers')
    assert isinstance(dispatcher.handlers, dict)
    assert len(dispatcher.handlers) == 0

def test_register_handler():
    """Test that handlers can be registered and stored properly."""
    dispatcher = EventDispatcher()
    
    # Define a simple handler function
    def handler(event):
        pass
    
    # Register the handler
    dispatcher.register(TestEvent, handler)
    
    # Verify registration was successful
    assert TestEvent in dispatcher.handlers
    assert handler in dispatcher.handlers[TestEvent]
    assert len(dispatcher.handlers[TestEvent]) == 1

def test_register_multiple_handlers():
    """Test that multiple handlers can be registered for the same event type."""
    dispatcher = EventDispatcher()
    
    # Define handler functions
    def handler1(event):
        pass
    
    def handler2(event):
        pass
    
    # Register both handlers
    dispatcher.register(TestEvent, handler1)
    dispatcher.register(TestEvent, handler2)
    
    # Verify both were registered
    assert len(dispatcher.handlers[TestEvent]) == 2
    assert handler1 in dispatcher.handlers[TestEvent]
    assert handler2 in dispatcher.handlers[TestEvent]

def test_register_same_handler_twice():
    """Test that registering the same handler twice doesn't duplicate it."""
    dispatcher = EventDispatcher()
    
    def handler(event):
        pass
    
    # Register the same handler twice
    dispatcher.register(TestEvent, handler)
    dispatcher.register(TestEvent, handler)
    
    # Verify it's only registered once
    assert len(dispatcher.handlers[TestEvent]) == 1

def test_unregister_handler():
    """Test that handlers can be unregistered correctly."""
    dispatcher = EventDispatcher()
    
    def handler(event):
        pass
    
    # Register and then unregister
    dispatcher.register(TestEvent, handler)
    dispatcher.unregister(TestEvent, handler)
    
    # Verify handler was removed
    assert TestEvent not in dispatcher.handlers

def test_unregister_with_multiple_handlers():
    """Test unregistering one handler while leaving others intact."""
    dispatcher = EventDispatcher()
    
    def handler1(event):
        pass
    
    def handler2(event):
        pass
    
    # Register both handlers
    dispatcher.register(TestEvent, handler1)
    dispatcher.register(TestEvent, handler2)
    
    # Unregister only handler1
    dispatcher.unregister(TestEvent, handler1)
    
    # Verify only handler1 was removed
    assert TestEvent in dispatcher.handlers
    assert handler1 not in dispatcher.handlers[TestEvent]
    assert handler2 in dispatcher.handlers[TestEvent]

def test_unregister_nonexistent_handler():
    """Test that unregistering a non-existent handler doesn't raise errors."""
    dispatcher = EventDispatcher()
    
    def handler(event):
        pass
    
    # Try to unregister a handler that was never registered
    dispatcher.unregister(TestEvent, handler)
    
    # No exception should be raised, and state should be unchanged
    assert TestEvent not in dispatcher.handlers

def test_dispatch_to_single_handler():
    """Test that events are properly dispatched to a single handler."""
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

def test_dispatch_to_multiple_handlers():
    """Test that events are dispatched to all registered handlers."""
    dispatcher = EventDispatcher()
    received_by_handler1 = []
    received_by_handler2 = []
    
    # Define two handlers
    def handler1(event):
        received_by_handler1.append(event)
    
    def handler2(event):
        received_by_handler2.append(event)
    
    # Register both handlers
    dispatcher.register(TestEvent, handler1)
    dispatcher.register(TestEvent, handler2)
    
    # Dispatch an event
    test_event = TestEvent(100, "test data")
    dispatcher.dispatch(test_event)
    
    # Verify both handlers received the event
    assert len(received_by_handler1) == 1
    assert len(received_by_handler2) == 1
    assert received_by_handler1[0] is test_event
    assert received_by_handler2[0] is test_event

def test_dispatch_only_to_matching_handlers():
    """Test that events are only dispatched to handlers for their specific type."""
    dispatcher = EventDispatcher()
    test_event_received = []
    other_event_received = []
    
    # Define handlers for different event types
    def test_handler(event):
        test_event_received.append(event)
    
    def other_handler(event):
        other_event_received.append(event)
    
    # Register handlers for different event types
    dispatcher.register(TestEvent, test_handler)
    dispatcher.register(OtherEvent, other_handler)
    
    # Dispatch a TestEvent
    test_event = TestEvent(100, "test data")
    dispatcher.dispatch(test_event)
    
    # Verify only the test_handler received it
    assert len(test_event_received) == 1
    assert len(other_event_received) == 0
    
    # Now dispatch an OtherEvent
    other_event = OtherEvent(200, "other info")
    dispatcher.dispatch(other_event)
    
    # Verify only the other_handler received it
    assert len(test_event_received) == 1  # Still just the one from before
    assert len(other_event_received) == 1

def test_no_handlers_for_event_type():
    """Test that dispatching an event with no handlers doesn't cause errors."""
    dispatcher = EventDispatcher()
    
    # No handlers registered
    test_event = TestEvent(100, "test data")
    
    # This should not raise any exceptions
    dispatcher.dispatch(test_event)