# delivery_sim/services/assignment_service.py
"""
Assignment Service - Driver-Entity Assignment with Priority Scoring

This service implements a hybrid assignment approach:
1. Immediate assignment for clear opportunities
2. Periodic global optimization using Hungarian algorithm

Core change: Replaced adjusted cost framework with multi-criteria priority scoring
for more principled assignment decisions.
"""

import numpy as np
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
    
    Uses priority scoring system to evaluate assignment opportunities based on:
    - Distance efficiency (infrastructure-aware)
    - Throughput optimization (orders per trip)
    - Fairness considerations (wait time)
    """
    
    def __init__(self, env, event_dispatcher, order_repository, driver_repository, 
                 pair_repository, delivery_unit_repository, priority_scorer, config):
        """
        Initialize the assignment service with its dependencies.
        
        Args:
            env: SimPy environment
            event_dispatcher: Central event dispatcher
            order_repository: Repository for orders
            driver_repository: Repository for drivers
            pair_repository: Repository for pairs
            delivery_unit_repository: Repository for delivery units
            priority_scorer: PriorityScorer instance for assignment evaluation
            config: Configuration object containing assignment parameters
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
        self.priority_scorer = priority_scorer
        self.config = config
        
        # Log service initialization with configuration details
        self.logger.info(f"[t={self.env.now:.2f}] AssignmentService initialized with priority scoring system")
        self.logger.info(f"[t={self.env.now:.2f}] Configuration: "
                        f"threshold={config.immediate_assignment_threshold}, "
                        f"periodic_interval={config.periodic_interval} min")
        
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
        
        Args:
            event: OrderCreatedEvent containing order_id
        """
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Handling OrderCreatedEvent for order {event.order_id}")
        
        # Get the order from repository
        order = self.order_repository.find_by_id(event.order_id)
        if order is None:
            self.logger.error(f"[t={self.env.now:.2f}] Order {event.order_id} not found in repository")
            return
        
        # Attempt immediate assignment
        self.attempt_immediate_assignment_from_delivery_entity(order)
    
    def handle_pair_created(self, event):
        """
        Handler for PairCreatedEvent in pairing mode.
        
        Args:
            event: PairCreatedEvent containing pair_id
        """
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Handling PairCreatedEvent for pair {event.pair_id}")
        
        # Get the pair from repository
        pair = self.pair_repository.find_by_id(event.pair_id)
        if pair is None:
            self.logger.error(f"[t={self.env.now:.2f}] Pair {event.pair_id} not found in repository")
            return
        
        # Attempt immediate assignment
        self.attempt_immediate_assignment_from_delivery_entity(pair)
    
    def handle_pairing_failed(self, event):
        """
        Handler for PairingFailedEvent - single order bypassed pairing.
        
        Args:
            event: PairingFailedEvent containing order_id
        """
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Handling PairingFailedEvent for order {event.order_id}")
        
        # Get the order from repository
        order = self.order_repository.find_by_id(event.order_id)
        if order is None:
            self.logger.error(f"[t={self.env.now:.2f}] Order {event.order_id} not found in repository")
            return
        
        # Attempt immediate assignment of the single order
        self.attempt_immediate_assignment_from_delivery_entity(order)
    
    def handle_driver_login(self, event):
        """
        Handler for DriverLoggedInEvent.
        
        Args:
            event: DriverLoggedInEvent containing driver_id
        """
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Handling DriverLoggedInEvent for driver {event.driver_id}")
        
        # Get the driver from repository
        driver = self.driver_repository.find_by_id(event.driver_id)
        if driver is None:
            self.logger.error(f"[t={self.env.now:.2f}] Driver {event.driver_id} not found in repository")
            return
        
        # Attempt immediate assignment from this new driver
        self.attempt_immediate_assignment_from_driver(driver)
    
    def handle_driver_available_for_assignment(self, event):
        """
        Handler for DriverAvailableForAssignmentEvent.
        
        Args:
            event: DriverAvailableForAssignmentEvent containing driver_id
        """
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Handling DriverAvailableForAssignmentEvent for driver {event.driver_id}")
        
        # Get the driver from repository
        driver = self.driver_repository.find_by_id(event.driver_id)
        if driver is None:
            self.logger.error(f"[t={self.env.now:.2f}] Driver {event.driver_id} not found in repository")
            return
        
        # Attempt immediate assignment from this available driver
        self.attempt_immediate_assignment_from_driver(driver)
    
    # ===== Assignment Operations =====
    
    def attempt_immediate_assignment_from_delivery_entity(self, delivery_entity):
        """
        Try to find a driver for a delivery entity immediately upon creation.
        
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
        best_driver, priority_score, score_components = self._find_best_match(delivery_entity, available_drivers)
        
        self.logger.debug(f"[t={self.env.now:.2f}] Best match for {entity_type} {entity_id}: "
                        f"driver {best_driver.driver_id} with priority score {priority_score:.2f} "
                        f"(distance: {score_components['distance_score']:.3f}, "
                        f"throughput: {score_components['throughput_score']:.3f}, "
                        f"fairness: {score_components['fairness_score']:.3f})")
        
        # Business decision: Check if assignment meets threshold criteria
        if priority_score >= self.config.immediate_assignment_threshold:
            # Create the assignment
            self.logger.info(f"[t={self.env.now:.2f}] Immediate assignment: {entity_type} {entity_id} meets threshold "
                           f"({priority_score:.2f} >= {self.config.immediate_assignment_threshold})")
            self._create_assignment(best_driver, delivery_entity, "immediate", score_components)
            return True
        else:
            # Business outcome: Score below immediate assignment threshold
            self.logger.info(f"[t={self.env.now:.2f}] Immediate assignment deferred: {entity_type} {entity_id} below threshold "
                           f"({priority_score:.2f} < {self.config.immediate_assignment_threshold})")
            return False
    
    def attempt_immediate_assignment_from_driver(self, driver):
        """
        Try to find the best delivery entity for a newly available driver.
        
        Args:
            driver: The driver to find work for
            
        Returns:
            bool: True if assignment succeeded, False otherwise
        """
        self.logger.debug(f"[t={self.env.now:.2f}] Attempting immediate assignment for driver {driver.driver_id}")
        
        # Business logic: Collect all available delivery entities
        waiting_entities = []
        waiting_entities.extend(self.order_repository.find_by_state(OrderState.CREATED))
        waiting_entities.extend(self.pair_repository.find_by_state(PairState.CREATED))
        
        if not waiting_entities:
            # Business outcome: No entities available for assignment
            self.logger.debug(f"[t={self.env.now:.2f}] No waiting entities for driver {driver.driver_id}")
            return False
        
        self.logger.debug(f"[t={self.env.now:.2f}] Found {len(waiting_entities)} waiting entities for driver {driver.driver_id}")
        
        # Find best entity for this driver
        best_entity, priority_score, score_components = self._find_best_match(driver, waiting_entities)
        
        entity_type = best_entity.entity_type
        entity_id = best_entity.order_id if entity_type == EntityType.ORDER else best_entity.pair_id
        
        self.logger.debug(f"[t={self.env.now:.2f}] Best match for driver {driver.driver_id}: "
                        f"{entity_type} {entity_id} with priority score {priority_score:.2f}")
        
        # Business decision: Check if assignment meets threshold criteria
        if priority_score >= self.config.immediate_assignment_threshold:
            # Create the assignment
            self.logger.info(f"[t={self.env.now:.2f}] Immediate assignment: driver {driver.driver_id} to {entity_type} {entity_id} meets threshold "
                           f"({priority_score:.2f} >= {self.config.immediate_assignment_threshold})")
            self._create_assignment(driver, best_entity, "immediate", score_components)
            return True
        else:
            # Business outcome: Score below immediate assignment threshold
            self.logger.info(f"[t={self.env.now:.2f}] Immediate assignment deferred: driver {driver.driver_id} to {entity_type} {entity_id} below threshold "
                           f"({priority_score:.2f} < {self.config.immediate_assignment_threshold})")
            return False
    
    # ===== Periodic Assignment Process =====
    
    def _periodic_assignment_process(self):
        """SimPy process for periodic global optimization."""
        interval = self.config.periodic_interval
        
        while True:
            # Wait for the next periodic assignment interval
            yield self.env.timeout(interval)
            
            # Perform periodic assignment
            self.perform_periodic_assignment()
    
    def perform_periodic_assignment(self):
        """
        Perform periodic global optimization using Hungarian algorithm.
        
        This method collects all waiting entities and available drivers,
        then uses optimization to find the globally optimal assignments.
        """
        self.logger.debug(f"[t={self.env.now:.2f}] Starting periodic assignment optimization")
        
        # Collect entities waiting for assignment
        waiting_entities = []
        waiting_entities.extend(self.order_repository.find_by_state(OrderState.CREATED))
        waiting_entities.extend(self.pair_repository.find_by_state(PairState.CREATED))
        
        # Collect available drivers
        available_drivers = self.driver_repository.find_available_drivers()
        
        self.logger.debug(f"[t={self.env.now:.2f}] Periodic optimization: {len(waiting_entities)} entities, {len(available_drivers)} drivers")
        
        # Check if optimization is needed
        if not waiting_entities or not available_drivers:
            self.logger.debug(f"[t={self.env.now:.2f}] Periodic optimization skipped: insufficient entities or drivers")
            return
        
        # Generate score matrix for optimization
        score_matrix = self._generate_score_matrix(waiting_entities, available_drivers)
        
        # Apply Hungarian algorithm (maximize scores)
        row_indices, col_indices = linear_sum_assignment(score_matrix, maximize=True)
        
        # Log the optimization results
        self.logger.info(f"[t={self.env.now:.2f}] Periodic optimization completed: {len(row_indices)} assignments identified")
        
        # Execute the optimal assignments
        assignments_created = 0
        for row, col in zip(row_indices, col_indices):
            entity = waiting_entities[row]
            driver = available_drivers[col]
            
            # Calculate full score information for record keeping
            priority_score, score_components = self.priority_scorer.calculate_priority_score(driver, entity)
            
            # Create the assignment (detailed logging happens inside this method)
            delivery_unit = self._create_assignment(driver, entity, "periodic", score_components)
            if delivery_unit:
                assignments_created += 1
        
        self.logger.info(f"[t={self.env.now:.2f}] Periodic optimization execution completed: {assignments_created} assignments created")
    
    def _generate_score_matrix(self, waiting_entities, available_drivers):
        """
        Generate score matrix for Hungarian algorithm optimization.
        
        Args:
            waiting_entities: List of unassigned orders and pairs
            available_drivers: List of available drivers
            
        Returns:
            numpy.ndarray: 2D matrix of priority scores
        """
        score_matrix = []
        
        for entity in waiting_entities:
            row = []
            entity_type = entity.entity_type
            entity_id = entity.order_id if entity_type == EntityType.ORDER else entity.pair_id
            
            for driver in available_drivers:
                priority_score, _ = self.priority_scorer.calculate_priority_score(driver, entity)
                row.append(priority_score)
                
                self.logger.debug(f"[t={self.env.now:.2f}] Score matrix entry: {entity_type} {entity_id} to driver {driver.driver_id} = {priority_score:.2f}")
            
            score_matrix.append(row)
        
        return np.array(score_matrix)
    
    def _find_best_match(self, fixed_entity, candidates):
        """
        Find the best matching entity from a list of candidates.
        
        Args:
            fixed_entity: Either a driver or delivery entity we're finding a match for
            candidates: List of potential matches (drivers or delivery entities)
        
        Returns:
            tuple: (best_match, best_score, best_components) or (None, 0.0, None) if no candidates
        """
        best_match = None
        best_priority_score = 0.0  # Start with lowest possible score
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
            
            # Calculate priority score
            priority_score, components = self.priority_scorer.calculate_priority_score(driver, entity)
            
            # Higher scores are better (opposite of cost minimization)
            if priority_score > best_priority_score:
                best_priority_score = priority_score
                best_match = candidate
                best_components = components
        
        return best_match, best_priority_score, best_components
    
    def _create_assignment(self, driver, entity, assignment_path, score_components):
        """
        Create a delivery unit and update entity states.
        
        Args:
            driver: The driver to assign
            entity: The order or pair to assign
            assignment_path: How this assignment was made ('immediate' or 'periodic')
            score_components: Dictionary with score calculation components
            
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
                        f"{entity_type} {entity_id}")
        
        delivery_unit = DeliveryUnit(entity, driver, self.env.now)
        delivery_unit.assignment_path = assignment_path
        
        # Record score components (replacing old cost storage)
        delivery_unit.assignment_scores = {
            "distance_score": score_components["distance_score"],
            "throughput_score": score_components["throughput_score"],
            "fairness_score": score_components["fairness_score"],
            "combined_score_0_1": score_components["combined_score_0_1"],
            "priority_score_0_100": score_components["combined_score_0_1"] * 100,
            "total_distance": score_components["total_distance"],
            "num_orders": score_components["num_orders"],
            "wait_time_minutes": score_components["wait_time_minutes"]
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
                       f"(priority score: {delivery_unit.assignment_scores['priority_score_0_100']:.2f}, "
                       f"distance: {score_components['total_distance']:.2f}km)")
        
        return delivery_unit