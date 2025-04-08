# Event-Driven Architecture for Food Delivery Simulation Systems

## Motivation: Why Adopt a New Architecture?

Complex simulation systems like food delivery models eventually face common challenges as they grow in sophistication:

1. **Increased Complexity**: As the simulation incorporates more advanced concepts (like multi-objective optimization), the interactions between components become harder to understand and trace.

2. **Data Flow Opacity**: It becomes difficult to follow how information moves through the system when components directly manipulate each other's state.

3. **Tight Coupling**: Changes in one component often require changes in many others, increasing maintenance burden.

4. **Implicit Dependencies**: Relationships between components become hidden in implementation details rather than explicit in the architecture.

5. **State Management Challenges**: As entities move through complex lifecycles, ensuring consistent state transitions becomes increasingly difficult.

An event-driven architecture addresses these challenges by:

- Making interactions explicit through events
- Centralizing state management
- Decoupling components
- Creating clear boundaries of responsibility
- Enabling traceable flows of information

The result is a simulation system that can grow in conceptual sophistication while remaining comprehensible and maintainable.

## Core Components of Event-Driven Architecture

### Entities

Entities are the core domain objects that represent key concepts in the simulation:

```python
class Order:
    def __init__(self, order_id, restaurant_location, customer_location, arrival_time):
        self.order_id = order_id
        self.restaurant_location = restaurant_location
        self.customer_location = customer_location
        self.arrival_time = arrival_time
        self.state = OrderState.CREATED
        self.pair = None
        self.delivery_unit = None
```

Key characteristics of entities:
- They have identity (typically through an ID)
- They have state (both in the general sense and specific state values)
- They represent meaningful domain concepts (Order, Driver, Pair)
- They contain data but typically delegate behavior to services

In a food delivery simulation, key entities include Orders, Drivers, Pairs, and DeliveryUnits.

### States

States represent the specific conditions that entities can be in at any given time:

```python
class OrderState:
    CREATED = "created"
    PAIRED = "paired"
    ASSIGNED = "assigned"
    PICKED_UP = "picked_up"
    DELIVERED = "delivered"
```

Key characteristics of states:
- They are typically implemented as enumerated values or constants
- They represent meaningful conditions in the domain
- They determine what actions are valid for an entity
- They change over time as events occur in the simulation

States are crucial for validation and for organizing domain logic. For example, only orders in the CREATED state can be paired, and only drivers in the AVAILABLE state can be assigned deliveries.

### Events

Events are messages that communicate that something has happened in the system:

```python
class Event:
    """Base class for all events"""
    def __init__(self, timestamp):
        self.timestamp = timestamp

class OrderCreatedEvent(Event):
    """Event for a new order entering the system"""
    def __init__(self, timestamp, order):
        super().__init__(timestamp)
        self.order = order
```

Key characteristics of events:
- They represent past occurrences, not future actions
- They are immutable (once created, they don't change)
- They contain all information handlers might need
- They follow a naming convention that indicates what happened
- They have no behavior, only data

Events come in two main types:
1. **Domain Events**: Represent significant business occurrences (OrderCreated, DeliveryAssigned)
2. **State Change Events**: Represent entity lifecycle transitions (OrderStateChanged)

### EventDispatcher

The EventDispatcher is the central coordination mechanism that connects event publishers with event handlers:

```python
class EventDispatcher:
    """Central hub for event broadcasting"""
    def __init__(self):
        self.handlers = {}  # Maps event types to lists of handler functions
        
    def register(self, event_type, handler):
        """Register a handler function for a specific event type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        
    def dispatch(self, event):
        """Send an event to all registered handlers"""
        event_type = type(event)
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                handler(event)
```

Key characteristics of the EventDispatcher:
- It maintains a registry of event types and their handlers
- It provides methods to register handlers and dispatch events
- It delivers events to all interested handlers
- It decouples event producers from event consumers
- It is typically a singleton (one instance shared by all components)

### Event Handlers

Event handlers are methods that respond to specific event types:

```python
def handle_order_created(self, event):
    """Handler for OrderCreatedEvent"""
    # Extract data from the event
    new_order = event.order
    
    # Call appropriate operations
    self.attempt_pairing(new_order)
```

Key characteristics of event handlers:
- They are registered with the EventDispatcher for specific event types
- They have a standard signature, taking an event parameter
- They extract data from events and delegate to operations
- They serve as entry points to service functionality 
- They focus on "what happened" and determine "what to do about it"

### Operations

Operations are methods that implement domain logic:

```python
def attempt_pairing(self, new_order):
    """Try to pair the new order with an existing order"""
    # Query operation: find candidate orders for pairing
    candidate_orders = self._find_pairing_candidates(new_order)
    
    # Query operation: evaluate and find the best pairing
    best_candidate = self._find_best_pairing_match(new_order, candidate_orders)
    
    # If we found a good match, create the pair
    if best_candidate:
        # Command operation: create a pair
        self._create_pair(new_order, best_candidate)
        return True
    
    return False
```

Key characteristics of operations:
- They implement business logic specific to the domain
- They can be classified as "commands" (change state) or "queries" (return information)
- They trigger state transitions and generate events
- They have domain-specific signatures
- They encapsulate the "how" of the system's behavior

### Services

Services are classes that group related operations and event handlers:

```python
class PairingService:
    """Service for pairing-related operations"""
    def __init__(self, event_dispatcher, order_repository):
        self.event_dispatcher = event_dispatcher
        self.order_repository = order_repository
        
        # Register for events this service cares about
        event_dispatcher.register(OrderCreatedEvent, self.handle_order_created)
    
    def handle_order_created(self, event):
        """Event handler that responds to new orders"""
        new_order = event.order
        self.attempt_pairing(new_order)
        
    def attempt_pairing(self, new_order):
        """Operation to pair a new order with existing orders"""
        # Implementation...
```

Key characteristics of services:
- They organize related operations and handlers
- They receive their dependencies through their constructor
- They register their event handlers during initialization
- They focus on specific aspects of domain functionality
- They coordinate activities in their domain area

## How Components Work Together to Drive the Simulation

The components of an event-driven architecture work together in a coordinated flow:

1. **External trigger or scheduled event** initiates action
2. **Service operation** executes business logic
3. **Entity state transitions** occur
4. **Events are dispatched** to announce what happened
5. **Event handlers** receive the events
6. **Handlers call operations** to process the events
7. **The cycle continues** with new events

Here's a concrete example of this flow in a food delivery simulation:

### New Order Flow

1. **External Trigger**: A new order enters the system
   ```python
   order_service.create_order(restaurant_location, customer_location)
   ```

2. **Operation Execution**: The `create_order` operation creates an Order entity
   ```python
   order = Order(order_id, restaurant_location, customer_location, arrival_time)
   ```

3. **Event Dispatch**: The operation announces the new order
   ```python
   event_dispatcher.dispatch(OrderCreatedEvent(timestamp, order))
   ```

4. **Event Handling**: The PairingService's handler receives the event
   ```python
   # In PairingService
   def handle_order_created(self, event):
       self.attempt_pairing(event.order)
   ```

5. **Operation Call**: The handler calls the pairing operation
   ```python
   # In PairingService
   def attempt_pairing(self, new_order):
       # Find best pairing...
       self._create_pair(new_order, best_candidate)
   ```

6. **State Transition**: The operation changes entity states
   ```python
   # In _create_pair method
   order1.transition_to(OrderState.PAIRED, self.event_dispatcher)
   order2.transition_to(OrderState.PAIRED, self.event_dispatcher)
   ```

7. **New Event Dispatch**: State changes and operation completion generate new events
   ```python
   # From transition_to method
   event_dispatcher.dispatch(OrderStateChangedEvent(timestamp, order, old_state, new_state))
   
   # From _create_pair method
   event_dispatcher.dispatch(PairCreatedEvent(timestamp, pair))
   ```

8. **Cycle Continues**: The AssignmentService handles the PairCreatedEvent
   ```python
   # In AssignmentService
   def handle_pair_created(self, event):
       self.attempt_immediate_assignment_for_entity(event.pair)
   ```

This cycle of events → handlers → operations → state changes → new events creates a traceable, decoupled flow through the system.

## State Transitions and Changes

State transitions are a critical part of simulation systems, as they represent how entities progress through their lifecycles:

```python
def transition_to(self, new_state, event_dispatcher=None):
    """Change the entity's state with validation"""
    # Define valid transitions
    valid_transitions = {
        OrderState.CREATED: [OrderState.PAIRED, OrderState.ASSIGNED],
        OrderState.PAIRED: [OrderState.ASSIGNED],
        OrderState.ASSIGNED: [OrderState.PICKED_UP],
        OrderState.PICKED_UP: [OrderState.DELIVERED]
    }
    
    # Validate the transition
    if self.state not in valid_transitions or new_state not in valid_transitions[self.state]:
        raise InvalidStateTransitionError(
            f"Cannot transition order {self.order_id} from {self.state} to {new_state}"
        )
    
    # Update state
    old_state = self.state
    self.state = new_state
    
    # Dispatch state change event
    if event_dispatcher:
        event_dispatcher.dispatch(OrderStateChangedEvent(
            timestamp=SimulationClock.now(),
            order=self,
            old_state=old_state,
            new_state=new_state
        ))
```

Key aspects of state management:

1. **Centralized Transition Method**: Each entity provides a `transition_to` method that centralizes state change logic.

2. **Validation**: The method enforces valid state transitions based on domain rules.

3. **Event Generation**: State changes generate events that announce the transition.

4. **Explicit Boundaries**: The validation creates clear boundaries around what can happen when.

State transitions typically occur within command operations when the business logic requires an entity to change its condition.

## Distinction Between Events and Operations

Understanding the difference between events and operations is fundamental to event-driven architecture:

### Events

- **Past Tense**: Represent things that have happened (OrderCreated, DeliveryCompleted)
- **Notification**: Announce that something occurred
- **Information Carriers**: Contain data about what happened
- **Immutable**: Don't change once created
- **Passive**: Have no behavior, just data

### Operations

- **Imperative or Interrogative**: Represent things to do or questions to answer (createOrder, findBestMatch)
- **Action**: Perform business logic
- **Behavior Implementers**: Contain the "how" of the system
- **Active**: Change state and generate events
- **Domain-Specific**: Have signatures and behavior tailored to their purpose

#### Query Operations vs. Command Operations

Operations come in two main types:

**Query Operations**:
- Return information without changing state
- Are idempotent (calling multiple times gives same result)
- Examples: findPairingCandidates, calculateAdjustedCost

```python
def _find_best_pairing_match(self, new_order, candidates):
    """Find the best order to pair with (Query operation)"""
    # This doesn't change any state, just returns information
    best_candidate = None
    best_score = float('inf')
    
    for candidate in candidates:
        # Calculate score logic...
        if score < best_score:
            best_score = score
            best_candidate = candidate
    
    return best_candidate
```

**Command Operations**:
- Change state in the system
- Generate events to announce changes
- Examples: createOrder, assignDelivery

```python
def _create_pair(self, order1, order2):
    """Create a pair from two orders (Command operation)"""
    # This changes state and generates events
    pair = Pair(order1, order2)
    
    # Update order states
    order1.pair = pair
    order2.pair = pair
    order1.transition_to(OrderState.PAIRED, self.event_dispatcher)
    order2.transition_to(OrderState.PAIRED, self.event_dispatcher)
    
    # Dispatch event
    self.event_dispatcher.dispatch(PairCreatedEvent(
        timestamp=SimulationClock.now(),
        pair=pair
    ))
```

### Attempt Operations

A special case worth noting is "attempt" operations, which try to do something but might not succeed:

```python
def attempt_immediate_assignment_from_driver(self, driver):
    """Try to find the best delivery unit for a driver"""
    # Find best unit based on adjusted cost
    # ...
    
    # Check if best unit meets threshold
    if best_cost <= self.threshold:
        # Success - perform assignment
        self.assign_delivery(driver, best_unit, best_cost)
        return True
    
    # Failure - no assignment made
    return False
```

These operations are still operations (not events), but they handle both success and failure paths. Depending on your system's needs, you might generate events for both outcomes or just for successes.

## Integration with SimPy

Since you prefer to continue using SimPy, here's how to integrate event-driven architecture with SimPy's infrastructure:

```python
import simpy

class DeliveryService:
    def __init__(self, env, event_dispatcher):
        self.env = env  # SimPy environment
        self.event_dispatcher = event_dispatcher
        
        # Register for events
        event_dispatcher.register(DeliveryAssignedEvent, self.handle_delivery_assigned)
    
    def handle_delivery_assigned(self, event):
        """Handle a new delivery assignment"""
        driver = event.driver
        delivery_unit = event.delivery_unit
        self.start_delivery(driver, delivery_unit)
    
    def start_delivery(self, driver, delivery_unit):
        """Begin the delivery process"""
        # Calculate travel time
        travel_time = self._estimate_travel_time(
            driver.location, 
            delivery_unit.pickup_locations[0]
        )
        
        # Update driver state
        driver.transition_to(DriverState.PICKING_UP, self.event_dispatcher)
        
        # Start a SimPy process for the delivery
        self.env.process(self._pickup_process(driver, delivery_unit, travel_time))
        
        # Dispatch event
        self.event_dispatcher.dispatch(DeliveryStartedEvent(
            timestamp=self.env.now,
            driver=driver,
            delivery_unit=delivery_unit,
            estimated_pickup_time=self.env.now + travel_time
        ))
    
    def _pickup_process(self, driver, delivery_unit, travel_time):
        """SimPy process for the pickup journey"""
        # Wait for travel time
        yield self.env.timeout(travel_time)
        
        # Driver has arrived at restaurant
        self.arrive_at_restaurant(driver, delivery_unit)
```

Key points for SimPy integration:

1. Use `env.now` instead of a custom clock for timestamps
2. Use `env.process()` to start SimPy processes
3. Use `yield env.timeout()` for time-based scheduling
4. Keep the event-driven pattern for communication between components
5. Use SimPy processes for activities that involve time delays

This approach leverages SimPy's mature process-based simulation capabilities while benefiting from the clarity and decoupling of event-driven architecture.

## Summary: The Value of Event-Driven Architecture

Event-driven architecture provides several key benefits for simulation systems:

1. **Explicit Interactions**: Components communicate through well-defined events rather than direct manipulation.

2. **Loose Coupling**: Components don't need detailed knowledge of each other, only the events they care about.

3. **Separation of Concerns**: Each component has a clear, focused responsibility in the system.

4. **Traceability**: The flow of events creates a visible trail of what happened in the system.

5. **Consistent State Management**: State transitions are centralized and validated.

6. **Extensibility**: New components can be added by registering for existing events.

7. **Testability**: Components can be tested in isolation with mock events.

For food delivery simulation specifically, this architecture helps manage the complex interactions between orders, drivers, pairs, and delivery processes while keeping the system comprehensible and maintainable even as the simulation model grows in sophistication.

By organizing the system around entities, states, events, and services, you create a structure that closely mirrors the domain concepts, making it easier to reason about the simulation and extend it as your research evolves.
