from scipy.optimize import linear_sum_assignment
from delivery_sim.entities.states import OrderState, DriverState, PairState
from delivery_sim.entities.delivery_unit import DeliveryUnit
from delivery_sim.events.order_events import OrderCreatedEvent
from delivery_sim.events.pair_events import PairCreatedEvent, PairingFailedEvent
from delivery_sim.events.driver_events import DriverLoggedInEvent, DriverAvailableForAssignmentEvent
from delivery_sim.events.delivery_unit_events import DeliveryUnitAssignedEvent
from delivery_sim.utils.location_utils import calculate_distance
from delivery_sim.utils.logging_system import get_logger
from delivery_sim.utils.entity_type_utils import EntityType

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
        # Get a logger instance specific to this component
        self.logger = get_logger("service.assignment")
        
        # Store dependencies
        self.env = env
        self.event_dispatcher = event_dispatcher
        self.order_repository = order_repository
        self.driver_repository = driver_repository
        self.pair_repository = pair_repository
        self.delivery_unit_repository = delivery_unit_repository
        self.config = config
        
        # Log service initialization with configuration details
        self.logger.info(f"[t={self.env.now:.2f}] AssignmentService initialized with configuration: "
                        f"threshold={config.immediate_assignment_threshold}, "
                        f"periodic_interval={config.periodic_interval} min, "
                        f"throughput_factor={config.throughput_factor}, "
                        f"age_factor={config.age_factor}")
        
        # Register event handlers based on pairing configuration
        if config.pairing_enabled:
            # In pair mode, listen for pairing outcomes
            self.logger.simulation_event(f"[t={self.env.now:.2f}] Registering handler for PairCreatedEvent")
            event_dispatcher.register(PairCreatedEvent, self.handle_pair_created)
            
            self.logger.simulation_event(f"[t={self.env.now:.2f}] Registering handler for PairingFailedEvent")
            event_dispatcher.register(PairingFailedEvent, self.handle_pairing_failed)
        else:
            # In single mode, orders directly go to assignment
            self.logger.simulation_event(f"[t={self.env.now:.2f}] Registering handler for OrderCreatedEvent")
            event_dispatcher.register(OrderCreatedEvent, self.handle_order_created)
        
        # Common events for both modes
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Registering handler for DriverLoggedInEvent")
        event_dispatcher.register(DriverLoggedInEvent, self.handle_driver_login)
        
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Registering handler for DriverAvailableForAssignmentEvent")
        event_dispatcher.register(DriverAvailableForAssignmentEvent, self.handle_driver_available_for_assignment)
        
        # Start the periodic assignment process
        self.logger.info(f"[t={self.env.now:.2f}] Starting periodic assignment process with interval {config.periodic_interval} minutes")
        self.process = env.process(self._periodic_assignment_process())
    
    # ===== Event Handlers (Entry Points) =====
    
    def handle_order_created(self, event):
        """
        Handler for OrderCreatedEvent in single mode.
        Validates the order and attempts immediate assignment if valid.
        """
        # Log event handling
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Handling OrderCreatedEvent for order {event.order_id}")
        
        # Validate order exists
        order_id = event.order_id
        order = self.order_repository.find_by_id(order_id)
        
        if not order:
            self.logger.validation(f"[t={self.env.now:.2f}] Order {order_id} not found, cannot attempt assignment")
            return
            
        # Pass validated entity to operation
        self.attempt_immediate_assignment_from_delivery_entity(order)
    
    def handle_pair_created(self, event):
        """
        Handler for PairCreatedEvent in pair mode.
        Validates the pair and attempts immediate assignment if valid.
        """
        # Log event handling
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Handling PairCreatedEvent for pair {event.pair_id}")
        
        # Validate pair exists
        pair_id = event.pair_id
        pair = self.pair_repository.find_by_id(pair_id)
        
        if not pair:
            self.logger.validation(f"[t={self.env.now:.2f}] Pair {pair_id} not found, cannot attempt assignment")
            return
            
        # Pass validated entity to operation
        self.attempt_immediate_assignment_from_delivery_entity(pair)
    
    def handle_pairing_failed(self, event):
        """
        Handler for PairingFailedEvent in pair mode.
        Validates the order and attempts immediate assignment if valid.
        """
        # Log event handling
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Handling PairingFailedEvent for order {event.order_id}")
        
        # Validate order exists
        order_id = event.order_id
        order = self.order_repository.find_by_id(order_id)
        
        if not order:
            self.logger.validation(f"[t={self.env.now:.2f}] Order {order_id} not found, cannot attempt assignment")
            return
            
        # Pass validated entity to operation
        self.attempt_immediate_assignment_from_delivery_entity(order)
    
    def handle_driver_login(self, event):
        """
        Handler for DriverLoggedInEvent.
        Validates the driver and attempts immediate assignment if valid.
        """
        # Log event handling
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Handling DriverLoggedInEvent for driver {event.driver_id}")
        
        # Validate driver exists
        driver_id = event.driver_id
        driver = self.driver_repository.find_by_id(driver_id)
        
        if not driver:
            self.logger.validation(f"[t={self.env.now:.2f}] Driver {driver_id} not found, cannot attempt assignment")
            return
            
        # Pass validated entity to operation
        self.attempt_immediate_assignment_from_driver(driver)
    
    def handle_driver_available_for_assignment(self, event):
        """
        Handler for DriverAvailableForAssignmentEvent(when a driver becomes available after delivery completion)
        Validates the driver and attempts immediate assignment if valid.
        """
        
        # Log event handling
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Handling DriverAvailableForAssignmentEvent for unit {event.delivery_unit_id} by driver {event.driver_id}")
        
        # Validate driver exists
        driver_id = event.driver_id
        driver = self.driver_repository.find_by_id(driver_id)
        
        if not driver:
            self.logger.validation(f"[t={self.env.now:.2f}] Driver {driver_id} not found, cannot attempt assignment")
            return
            
        # Pass validated entity to operation
        self.attempt_immediate_assignment_from_driver(driver)
    
    # ===== Operations =====
    
    def attempt_immediate_assignment_from_delivery_entity(self, delivery_entity):
        """
        Try to find an available driver for a delivery entity.
        
        This operation implements the business logic for immediate assignment decisions,
        determining if a clear opportunity exists to assign a driver immediately.
        
        Args:
            delivery_entity: The order or pair to assign
            
        Returns:
            bool: True if assignment succeeded, False otherwise
        """
        # Get entity type directly from the entity
        entity_type = delivery_entity.entity_type
        entity_id = delivery_entity.order_id if entity_type == EntityType.ORDER else delivery_entity.pair_id
        
        self.logger.debug(f"[t={self.env.now:.2f}] Attempting immediate assignment for {entity_type} {entity_id}")
    
        
        # Business logic: Check for available drivers
        available_drivers = self.driver_repository.find_available_drivers()
        if not available_drivers:
            # Business outcome: No drivers available to make assignment
            self.logger.debug(f"[t={self.env.now:.2f}] No available drivers for {entity_type} {entity_id}, assignment deferred")
            return False
        
        self.logger.debug(f"[t={self.env.now:.2f}] Found {len(available_drivers)} available drivers for {entity_type} {entity_id}")
        
        # Find best driver for this entity
        best_driver, adjusted_cost, cost_components = self._find_best_match(delivery_entity, available_drivers)
        
        self.logger.debug(f"[t={self.env.now:.2f}] Best match for {entity_type} {entity_id}: "
                        f"driver {best_driver.driver_id} with adjusted cost {adjusted_cost:.2f} "
                        f"(base: {cost_components['base_cost']:.2f}, throughput: {cost_components['throughput_component']:.2f}, "
                        f"age: {cost_components['age_discount']:.2f})")
        
        # Business decision: Check if assignment meets threshold criteria
        if adjusted_cost <= self.config.immediate_assignment_threshold:
            # Create the assignment
            self.logger.info(f"[t={self.env.now:.2f}] Immediate assignment: {entity_type} {entity_id} meets threshold "
                           f"({adjusted_cost:.2f} <= {self.config.immediate_assignment_threshold})")
            self._create_assignment(best_driver, delivery_entity, "immediate", cost_components)
            return True
        else:
            # Business outcome: Cost exceeds immediate assignment threshold
            self.logger.info(f"[t={self.env.now:.2f}] Immediate assignment deferred: {entity_type} {entity_id} exceeds threshold "
                           f"({adjusted_cost:.2f} > {self.config.immediate_assignment_threshold})")
            return False
    
    def attempt_immediate_assignment_from_driver(self, driver):
        """
        Try to find the best delivery entity for a newly available driver.
        
        This operation implements the business logic for immediate assignment decisions
        from the driver perspective, determining if there's a clear opportunity to 
        assign the driver to a waiting order or pair.
        
        Args:
            driver: The driver who became available
            
        Returns:
            bool: True if assignment succeeded, False otherwise
        """
        self.logger.debug(f"[t={self.env.now:.2f}] Attempting immediate assignment for driver {driver.driver_id}")
        
        # Get unassigned delivery entities
        unassigned_orders = self.order_repository.find_unassigned_orders()
        unassigned_pairs = self.pair_repository.find_unassigned_pairs()
        
        # Combine into a single list of candidates
        delivery_candidates = unassigned_orders + unassigned_pairs
        
        # Business logic: Check if there are waiting delivery entities
        if not delivery_candidates:
            # Business outcome: No waiting delivery entities to assign
            self.logger.debug(f"[t={self.env.now:.2f}] No waiting delivery entities for driver {driver.driver_id}, assignment deferred")
            return False
        
        self.logger.debug(f"[t={self.env.now:.2f}] Found {len(delivery_candidates)} waiting delivery entities for driver {driver.driver_id}")
        
        # Find best entity for this driver
        best_entity, adjusted_cost, cost_components = self._find_best_match(driver, delivery_candidates)
        
        # Get entity type directly from the entity
        entity_type = best_entity.entity_type 
        entity_id = best_entity.order_id if entity_type == EntityType.ORDER else best_entity.pair_id

        # For logging messages where we still use string representation
        entity_type_str = "order" if entity_type == EntityType.ORDER else "pair"

        self.logger.debug(f"[t={self.env.now:.2f}] Best match for driver {driver.driver_id}: "
                        f"{entity_type_str} {entity_id} with adjusted cost {adjusted_cost:.2f} "
                        f"(base: {cost_components['base_cost']:.2f}, throughput: {cost_components['throughput_component']:.2f}, "
                        f"age: {cost_components['age_discount']:.2f})")
        
        # Business decision: Check if assignment meets threshold criteria
        if adjusted_cost <= self.config.immediate_assignment_threshold:
            # Create the assignment
            self.logger.info(f"[t={self.env.now:.2f}] Immediate assignment: driver {driver.driver_id} to {entity_type} {entity_id} meets threshold "
                           f"({adjusted_cost:.2f} <= {self.config.immediate_assignment_threshold})")
            self._create_assignment(driver, best_entity, "immediate", cost_components)
            return True
        else:
            # Business outcome: Cost exceeds immediate assignment threshold
            self.logger.info(f"[t={self.env.now:.2f}] Immediate assignment deferred: driver {driver.driver_id} to {entity_type} {entity_id} exceeds threshold "
                           f"({adjusted_cost:.2f} > {self.config.immediate_assignment_threshold})")
            return False
    
    def _find_best_match(self, fixed_entity, candidates):
        """
        Find the best matching entity from a list of candidates.
        
        Args:
            fixed_entity: Either a driver or delivery entity we're finding a match for
            candidates: List of potential matches (drivers or delivery entities)
        
        Returns:
            tuple: (best_match, best_cost, best_components) or (None, float('inf'), None) if no candidates
        """
        best_match = None
        best_adjusted_cost = float('inf')
        best_components = None
        
        # Determine entity type for role assignment
        fixed_entity_type = fixed_entity.entity_type
        is_driver = fixed_entity_type == EntityType.DRIVER
        
        for candidate in candidates:
            # Determine which is the driver and which is the entity
            if is_driver:
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
        
        Args:
            driver: The driver to assign
            entity: The order or pair to assign
            assignment_path: How this assignment was made ('immediate' or 'periodic')
            cost_components: Dictionary with cost calculation components
            
        Returns:
            DeliveryUnit: The created delivery unit
        """
        # Critical validation - ensure driver is still available
        if driver.state != DriverState.AVAILABLE:
            self.logger.validation(f"[t={self.env.now:.2f}] Critical error: Driver {driver.driver_id} not available when creating assignment")
            return None
        
        # Get entity type once
        entity_type = entity.entity_type
        entity_id = entity.order_id if entity_type == EntityType.ORDER else entity.pair_id
            
        # Create delivery unit
        self.logger.debug(f"[t={self.env.now:.2f}] Creating delivery unit for assignment of driver {driver.driver_id} to "
                        f"{entity_type} "  # Use entity_type directly
                        f"{entity_id}")    # Use entity_id already computed
        
        delivery_unit = DeliveryUnit(entity, driver, self.env.now)
        delivery_unit.assignment_path = assignment_path
        
        # Record cost components
        delivery_unit.assignment_costs = {
            "base_cost": cost_components["base_cost"],
            "throughput_factor": self.config.throughput_factor,
            "throughput_discount": cost_components["throughput_component"],
            "age_factor": self.config.age_factor,
            "age_discount": cost_components["age_discount"],
            "adjusted_cost": cost_components["base_cost"] - 
                             cost_components["throughput_component"] - 
                             cost_components["age_discount"]
        }
        
        # Add to repository
        self.delivery_unit_repository.add(delivery_unit)
        
        # Update entity state based on type
        if entity_type == EntityType.ORDER:
            entity.transition_to(OrderState.ASSIGNED, self.event_dispatcher, self.env)
            entity.delivery_unit = delivery_unit
            self.logger.debug(f"[t={self.env.now:.2f}] Updated order {entity.order_id} state to ASSIGNED")
        else:  # Must be PAIR
            entity.transition_to(PairState.ASSIGNED, self.event_dispatcher, self.env)
            entity.delivery_unit = delivery_unit
            # Also update constituent orders
            entity.order1.transition_to(OrderState.ASSIGNED, self.event_dispatcher, self.env)
            entity.order1.delivery_unit = delivery_unit
            entity.order2.transition_to(OrderState.ASSIGNED, self.event_dispatcher, self.env)
            entity.order2.delivery_unit = delivery_unit
            self.logger.debug(f"[t={self.env.now:.2f}] Updated pair {entity.pair_id} state to ASSIGNED")
    
        # Update driver state
        driver.transition_to(DriverState.DELIVERING, self.event_dispatcher, self.env)
        driver.current_delivery_unit = delivery_unit
        self.logger.debug(f"[t={self.env.now:.2f}] Updated driver {driver.driver_id} state to DELIVERING")
        
        # Dispatch event
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Dispatching DeliveryUnitAssignedEvent for unit {delivery_unit.unit_id}")
        self.event_dispatcher.dispatch(DeliveryUnitAssignedEvent(
            timestamp=self.env.now,
            delivery_unit_id=delivery_unit.unit_id,
            entity_type=entity_type,  
            entity_id=entity_id,
            driver_id=driver.driver_id
        ))
        
        # Log assignment completion
        self.logger.info(f"[t={self.env.now:.2f}] Created {assignment_path} assignment: "
                       f"Driver {driver.driver_id} assigned to {entity_type} {entity_id} "
                       f"(base cost: {cost_components['base_cost']:.2f}, "
                       f"adjusted cost: {delivery_unit.assignment_costs['adjusted_cost']:.2f})")
        
        return delivery_unit
    
    def calculate_adjusted_cost(self, driver, entity):
        """
        Calculate adjusted cost using weighted objective function.
        
        Adjusted Cost = base_cost - throughput_factor * num_orders - age_factor * age
        
        Args:
            driver: The driver being evaluated
            entity: The order or pair being evaluated
            
        Returns:
            tuple: (adjusted_cost, components_dictionary)
        """
        # Get entity type once
        entity_type = entity.entity_type

        # Calculate base delivery cost
        base_cost = self.calculate_base_delivery_cost(driver, entity)
        
        # Calculate throughput component
        num_orders = 2 if entity_type == EntityType.PAIR else 1
        throughput_component = self.config.throughput_factor * num_orders
        
        # Calculate age component (fairness)
        if entity_type == EntityType.PAIR:
            arrival_time = min(entity.order1.arrival_time, entity.order2.arrival_time)
        else:  # Must be ORDER
            arrival_time = entity.arrival_time

        age_minutes = self.env.now - arrival_time
        age_discount = self.config.age_factor * age_minutes
        
        # Calculate adjusted cost
        adjusted_cost = base_cost - throughput_component - age_discount
        
        self.logger.debug(f"[t={self.env.now:.2f}] Calculated adjusted cost: {adjusted_cost:.2f} "
                        f"(base: {base_cost:.2f}, throughput: {throughput_component:.2f}, age: {age_discount:.2f})")
        
        # Return cost and components for logging
        components = {
            "base_cost": base_cost,
            "num_orders": num_orders,
            "throughput_component": throughput_component,
            "age_minutes": age_minutes,
            "age_discount": age_discount
        }
        
        return adjusted_cost, components
    
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
        entity_type = delivery_entity.entity_type
        
        if entity_type == EntityType.ORDER:
            cost = calculate_distance(driver.location, delivery_entity.restaurant_location) + \
                calculate_distance(delivery_entity.restaurant_location, delivery_entity.customer_location)
                
            self.logger.debug(f"[t={self.env.now:.2f}] Calculated base cost for single order {delivery_entity.order_id}: {cost:.2f}")
            return cost
        else:  # Must be PAIR
            # First leg is from driver to first pickup location
            cost = calculate_distance(driver.location, delivery_entity.optimal_sequence[0]) + \
                delivery_entity.optimal_cost
                
            self.logger.debug(f"[t={self.env.now:.2f}] Calculated base cost for pair {delivery_entity.pair_id}: {cost:.2f}")
            return cost
    
    def _periodic_assignment_process(self):
        """SimPy process that runs the periodic global optimization."""
        # Counter for epoch logging
        epoch_count = 0
        
        while True:
            yield self.env.timeout(self.config.periodic_interval)
            
            epoch_count += 1
            self.logger.info(f"[t={self.env.now:.2f}] Starting periodic assignment optimization (epoch {epoch_count})")
            self.perform_periodic_assignment(epoch_count)
    
    def perform_periodic_assignment(self, epoch_count=0):
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
        self.logger.info(f"[t={self.env.now:.2f}] Periodic optimization: "
                       f"{len(waiting_entities)} waiting delivery entities, {len(available_drivers)} available drivers")
        
        # Exit if we don't have both drivers and waiting entities
        if not waiting_entities:
            self.logger.info(f"[t={self.env.now:.2f}] Periodic optimization skipped: No waiting delivery entities to assign")
            return
        
        if not available_drivers:
            self.logger.info(f"[t={self.env.now:.2f}] Periodic optimization skipped: No available drivers")
            return
        
        # Log information about potential assignment imbalance
        if len(waiting_entities) > len(available_drivers):
            self.logger.debug(f"[t={self.env.now:.2f}] Assignment imbalance: {len(waiting_entities) - len(available_drivers)} "
                            f"waiting delivery entities will remain unassigned after periodic optimization")
        elif len(available_drivers) > len(waiting_entities):
            self.logger.debug(f"[t={self.env.now:.2f}] Assignment imbalance: {len(available_drivers) - len(waiting_entities)} "
                            f"drivers will remain unassigned after periodic optimization")
        else:
            self.logger.debug(f"[t={self.env.now:.2f}] Balanced assignment: Equal number of waiting delivery entities and available drivers")
        
        # Generate cost matrix
        self.logger.debug(f"[t={self.env.now:.2f}] Generating cost matrix for {len(waiting_entities)} entities and {len(available_drivers)} drivers")
        cost_matrix = self._generate_cost_matrix(waiting_entities, available_drivers)
        
        # Use Hungarian algorithm to find optimal assignment
        self.logger.debug(f"[t={self.env.now:.2f}] Running Hungarian algorithm to find optimal assignment")
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Log the optimization results
        self.logger.info(f"[t={self.env.now:.2f}] Periodic optimization completed: {len(row_indices)} assignments identified")
        
        # Execute the optimal assignments
        assignments_created = 0
        for row, col in zip(row_indices, col_indices):
            entity = waiting_entities[row]
            driver = available_drivers[col]
            
            # Calculate full cost information for record keeping
            _, cost_components = self.calculate_adjusted_cost(driver, entity)
            
            # Create the assignment (detailed logging happens inside this method)
            delivery_unit = self._create_assignment(driver, entity, "periodic", cost_components)
            if delivery_unit:
                assignments_created += 1
        
        self.logger.info(f"[t={self.env.now:.2f}] Periodic optimization execution completed: {assignments_created} assignments created")
    
    def _generate_cost_matrix(self, waiting_entities, available_drivers):
        """
        Generate cost matrix for Hungarian algorithm.
        
        Args:
            waiting_entities: List of unassigned orders and pairs
            available_drivers: List of available drivers
            
        Returns:
            list: 2D matrix of adjusted costs
        """
        cost_matrix = []
        
        for entity in waiting_entities:
            row = []
            entity_type = entity.entity_type
            entity_id = entity.order_id if entity_type == EntityType.ORDER else entity.pair_id
            
            for driver in available_drivers:
                adjusted_cost, _ = self.calculate_adjusted_cost(driver, entity)
                row.append(adjusted_cost)
                
                self.logger.debug(f"[t={self.env.now:.2f}] Cost matrix entry: {entity_type} {entity_id} to driver {driver.driver_id} = {adjusted_cost:.2f}")
            
            cost_matrix.append(row)
        
        return cost_matrix