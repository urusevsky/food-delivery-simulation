from scipy.optimize import linear_sum_assignment
from delivery_sim.entities.states import OrderState, DriverState, PairState
from delivery_sim.entities.delivery_unit import DeliveryUnit
from delivery_sim.events.order_events import OrderCreatedEvent
from delivery_sim.events.pair_events import PairCreatedEvent, PairingFailedEvent
from delivery_sim.events.driver_events import DriverLoggedInEvent
from delivery_sim.events.delivery_unit_events import DeliveryUnitCompletedEvent, DeliveryUnitAssignedEvent

class AssignmentService:
    """
    Service responsible for assigning drivers to delivery entities (orders or pairs).
    
    This service implements a hybrid assignment approach combining:
    1. Immediate assignment for clear opportunities
    2. Periodic global optimization for complex decisions
    """
    
    def __init__(self, env, event_dispatcher, order_repository, driver_repository, 
                 pair_repository, delivery_unit_repository, config):
        """
        Initialize the assignment service with its dependencies.
        """
        self.env = env
        self.event_dispatcher = event_dispatcher
        self.order_repository = order_repository
        self.driver_repository = driver_repository
        self.pair_repository = pair_repository
        self.delivery_unit_repository = delivery_unit_repository
        self.config = config
        
        # Register event handlers based on pairing configuration
        if config.pairing_enabled:
            # In pair mode, listen for pairing outcomes
            event_dispatcher.register(PairCreatedEvent, self.handle_pair_created)
            event_dispatcher.register(PairingFailedEvent, self.handle_pairing_failed)
        else:
            # In single mode, orders directly go to assignment
            event_dispatcher.register(OrderCreatedEvent, self.handle_order_created)
        
        # Common events for both modes
        event_dispatcher.register(DriverLoggedInEvent, self.handle_driver_login)
        event_dispatcher.register(DeliveryUnitCompletedEvent, self.handle_delivery_completed)
        
        # Start the periodic assignment process
        self.process = env.process(self._periodic_assignment_process())
    
    # ===== Event Handlers =====
    
    def handle_order_created(self, event):
        """
        Handler for OrderCreatedEvent in single mode.
        Attempts immediate assignment for the new order.
        """
        order_id = event.order_id
        self.attempt_immediate_assignment_from_delivery_entity(order_id, "order")
    
    def handle_pair_created(self, event):
        """
        Handler for PairCreatedEvent in pair mode.
        Attempts immediate assignment for the newly formed pair.
        """
        pair_id = event.pair_id
        self.attempt_immediate_assignment_from_delivery_entity(pair_id, "pair")
    
    def handle_pairing_failed(self, event):
        """
        Handler for PairingFailedEvent in pair mode.
        Attempts immediate assignment for the order that failed to pair.
        """
        order_id = event.order_id
        self.attempt_immediate_assignment_from_delivery_entity(order_id, "order")
    
    def handle_driver_login(self, event):
        """
        Handler for DriverLoggedInEvent.
        Attempts immediate assignment for the newly available driver.
        """
        driver_id = event.driver_id
        self.attempt_immediate_assignment_from_driver(driver_id)
    
    def handle_delivery_completed(self, event):
        """
        Handler for DeliveryUnitCompletedEvent.
        Attempts immediate assignment for the driver who just completed a delivery.
        """
        driver_id = event.driver_id
        self.attempt_immediate_assignment_from_driver(driver_id)
    
    # ===== Operations =====
    
    def attempt_immediate_assignment_from_delivery_entity(self, delivery_entity_id, entity_type):
        """
        Try to find an available driver for a delivery entity.
        
        This method implements the business logic for immediate assignment decisions,
        determining if a clear opportunity exists to assign a driver immediately.
        
        Args:
            delivery_entity_id: ID of the order or pair to assign
            entity_type: Type of entity ("order" or "pair")
            
        Returns:
            bool: True if assignment succeeded, False otherwise
        """
        # Get the entity from repository 
        # (assuming validation happens in test layer)
        delivery_entity = (self.order_repository.find_by_id(delivery_entity_id) 
                        if entity_type == "order" 
                        else self.pair_repository.find_by_id(delivery_entity_id))
        
        # Business logic: Check for available drivers
        available_drivers = self.driver_repository.find_available_drivers()
        if not available_drivers:
            # Business outcome: No drivers available to make assignment
            return False
        
        # Find best driver for this entity
        best_driver, adjusted_cost, cost_components = self._find_best_match(delivery_entity, available_drivers)
        
        # Business decision: Check if assignment meets threshold criteria
        if adjusted_cost <= self.config.immediate_assignment_threshold:
            # Create the assignment
            self._create_assignment(best_driver, delivery_entity, "immediate", cost_components)
            return True
        else:
            # Business outcome: Cost exceeds immediate assignment threshold
            return False
    
    def attempt_immediate_assignment_from_driver(self, driver_id):
        """
        Try to find the best delivery entity for a newly available driver.
        
        This method implements the business logic for immediate assignment decisions
        from the driver perspective, determining if there's a clear opportunity to 
        assign the driver to a waiting order or pair.
        
        Args:
            driver_id: ID of the driver who became available
            
        Returns:
            bool: True if assignment succeeded, False otherwise
        """
        # Get the driver from repository
        # (assuming validation happens in test layer)
        driver = self.driver_repository.find_by_id(driver_id)
        
        # Get unassigned delivery entities
        unassigned_orders = self.order_repository.find_unassigned_orders()
        unassigned_pairs = self.pair_repository.find_unassigned_pairs()
        
        # Combine into a single list of candidates
        delivery_candidates = unassigned_orders + unassigned_pairs
        
        # Business logic: Check if there are waiting delivery entities
        if not delivery_candidates:
            # Business outcome: No waiting delivery entities to assign
            return False
        
        # Find best entity for this driver
        best_entity, adjusted_cost, cost_components = self._find_best_match(driver, delivery_candidates)
        
        # Business decision: Check if assignment meets threshold criteria
        if adjusted_cost <= self.config.immediate_assignment_threshold:
            # Create the assignment
            self._create_assignment(driver, best_entity, "immediate", cost_components)
            return True
        else:
            # Business outcome: Cost exceeds immediate assignment threshold
            return False
    
    def _find_best_match(self, fixed_entity, candidates):
        """
        Find the best matching entity from a list of candidates.
        """
        best_match = None
        best_adjusted_cost = float('inf')
        best_components = None
        
        for candidate in candidates:
            # Determine which is the driver and which is the entity
            if hasattr(fixed_entity, 'driver_id'):
                driver = fixed_entity
                entity = candidate
            else:
                driver = candidate
                entity = fixed_entity
            
            # Calculate adjusted cost
            adjusted_cost, components = self.calculate_adjusted_cost(driver, entity)
            
            if adjusted_cost < best_adjusted_cost:
                best_adjusted_cost = adjusted_cost
                best_match = candidate
                best_components = components
        
        return best_match, best_adjusted_cost, best_components
    
    def _create_assignment(self, driver, entity, assignment_path, cost_components):
        """
        Create a delivery unit and update entity states.
        """
        # Create delivery unit
        delivery_unit = DeliveryUnit(entity, driver, self.env.now)
        delivery_unit.assignment_path = assignment_path
        
        # Record cost components
        delivery_unit.assignment_costs = {
            "base_cost": cost_components["base_cost"],
            "throughput_factor": self.config.throughput_factor,
            "throughput_discount": cost_components["throughput_discount"],
            "age_factor": self.config.age_factor,
            "age_discount": cost_components["age_discount"],
            "adjusted_cost": cost_components["base_cost"] - 
                             cost_components["throughput_discount"] - 
                             cost_components["age_discount"]
        }
        
        # Add to repository
        self.delivery_unit_repository.add(delivery_unit)
        
        # Update entity state
        if hasattr(entity, 'order_id'):  # It's an order
            entity.transition_to(OrderState.ASSIGNED, self.event_dispatcher, self.env)
            entity.delivery_unit = delivery_unit
        else:  # It's a pair
            entity.transition_to(PairState.ASSIGNED, self.event_dispatcher, self.env)
            entity.delivery_unit = delivery_unit
            # Also update constituent orders
            entity.order1.transition_to(OrderState.ASSIGNED, self.event_dispatcher, self.env)
            entity.order1.delivery_unit = delivery_unit
            entity.order2.transition_to(OrderState.ASSIGNED, self.event_dispatcher, self.env)
            entity.order2.delivery_unit = delivery_unit
        
        # Update driver state
        driver.transition_to(DriverState.DELIVERING, self.event_dispatcher, self.env)
        driver.current_delivery_unit = delivery_unit
        
        # Dispatch event
        entity_type = "order" if hasattr(entity, 'order_id') else "pair"
        entity_id = entity.order_id if entity_type == "order" else entity.pair_id
        
        self.event_dispatcher.dispatch(DeliveryUnitAssignedEvent(
            timestamp=self.env.now,
            delivery_unit_id=delivery_unit.unit_id,
            entity_type=entity_type,
            entity_id=entity_id,
            driver_id=driver.driver_id
        ))
        
        # Log for debugging
        print(f"Created {assignment_path} assignment at time {self.env.now}: "
              f"Driver {driver.driver_id} assigned to "
              f"{entity_type.capitalize()} {entity_id}")
        
        return delivery_unit
    
    def calculate_base_delivery_cost(self, driver, delivery_entity):
        """
        Calculate the base travel cost for a potential delivery assignment.
        
        This represents the actual distance/time the driver would need to travel:
        - For single orders: driver location → restaurant → customer
        - For pairs: driver location → first location in optimal sequence → rest of sequence
        
        Args:
            driver: The driver being evaluated
            delivery_entity: The order or pair being evaluated
            
        Returns:
            float: Total travel distance/time cost
        """
        if hasattr(delivery_entity, 'order_id'):  # It's an order
            return self._distance(driver.location, delivery_entity.restaurant_location) + \
                self._distance(delivery_entity.restaurant_location, delivery_entity.customer_location)
        else:  # It's a pair
            # First leg is from driver to first pickup location
            return self._distance(driver.location, delivery_entity.optimal_sequence[0]) + \
                delivery_entity.optimal_cost
    
    def calculate_adjusted_cost(self, driver, entity):
        """
        Calculate adjusted cost using weighted objective function.
        
        Adjusted Cost = base_cost - throughput_factor * num_orders - age_factor * age
        """
        # Calculate base delivery cost
        base_cost = self.calculate_base_delivery_cost(driver, entity)
        
        # Calculate throughput component
        num_orders = 2 if hasattr(entity, 'pair_id') else 1
        throughput_discount = self.config.throughput_factor * num_orders
        
        # Calculate age component (fairness)
        if hasattr(entity, 'pair_id'):  # It's a pair
            arrival_time = min(entity.order1.arrival_time, entity.order2.arrival_time)
        else:  # It's an order
            arrival_time = entity.arrival_time
        
        age_minutes = self.env.now - arrival_time
        age_discount = self.config.age_factor * age_minutes
        
        # Calculate adjusted cost
        adjusted_cost = base_cost - throughput_discount - age_discount
        
        # Return cost and components for logging
        components = {
            "base_cost": base_cost,
            "num_orders": num_orders,
            "throughput_discount": throughput_discount,
            "age_minutes": age_minutes,
            "age_discount": age_discount
        }
        
        return adjusted_cost, components
    
    def _distance(self, loc1, loc2):
        """
        Calculate Euclidean distance between two locations.
        """
        return ((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)**0.5
    
    def _periodic_assignment_process(self):
        """SimPy process that runs the periodic global optimization."""
        while True:
            yield self.env.timeout(self.config.periodic_interval)
            self.perform_periodic_assignment()
    
    def perform_periodic_assignment(self):
        """
        Apply global optimization to find optimal assignments for all waiting entities.
        
        This method implements the periodic assignment path, which considers all 
        available drivers and waiting delivery entities to find the system-wide 
        optimal assignment pattern.
        """
        # Get unassigned delivery entities
        unassigned_orders = self.order_repository.find_unassigned_orders()
        unassigned_pairs = self.pair_repository.find_unassigned_pairs()
        
        # Combine into a single list
        waiting_entities = unassigned_orders + unassigned_pairs
        
        # Get available drivers
        available_drivers = self.driver_repository.find_available_drivers()
        
        # Log periodic optimization attempt and system state
        print(f"Periodic optimization at time {self.env.now}: "
            f"{len(waiting_entities)} waiting delivery entities, {len(available_drivers)} available drivers")
        
        # Exit if we don't have both drivers and waiting entities
        if not waiting_entities:
            print(f"Periodic optimization skipped: No waiting delivery entities to assign")
            return
        
        if not available_drivers:
            print(f"Periodic optimization skipped: No available drivers")
            return
        
        # Log information about potential assignment imbalance
        if len(waiting_entities) > len(available_drivers):
            print(f"Assignment imbalance: {len(waiting_entities) - len(available_drivers)} "
                f"waiting delivery entities will remain unassigned after periodic optimization")
        elif len(available_drivers) > len(waiting_entities):
            print(f"Assignment imbalance: {len(available_drivers) - len(waiting_entities)} "
                f"drivers will remain unassigned after periodic optimization")
        else:
            print(f"Balanced assignment: Equal number of waiting delivery entities and available drivers")
        
        # Generate cost matrix
        cost_matrix = self._generate_cost_matrix(waiting_entities, available_drivers)
        
        # Use Hungarian algorithm to find optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Log the optimization results
        print(f"Periodic optimization completed: {len(row_indices)} assignments created")
        
        # Execute the optimal assignments
        for row, col in zip(row_indices, col_indices):
            entity = waiting_entities[row]
            driver = available_drivers[col]
            
            # Calculate full cost information for record keeping
            _, cost_components = self.calculate_adjusted_cost(driver, entity)
            
            # Create the assignment (detailed logging happens inside this method)
            self._create_assignment(driver, entity, "periodic", cost_components)
    
    def _generate_cost_matrix(self, waiting_entities, available_drivers):
        """
        Generate cost matrix for Hungarian algorithm.
        """
        cost_matrix = []
        
        for entity in waiting_entities:
            row = []
            for driver in available_drivers:
                adjusted_cost, _ = self.calculate_adjusted_cost(driver, entity)
                row.append(adjusted_cost)
            cost_matrix.append(row)
        
        return cost_matrix
    
