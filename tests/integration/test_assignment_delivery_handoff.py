# tests/integration/test_assignment_delivery_handoff.py
"""
Enhanced Assignment-Delivery Handoff Integration Tests

This file tests the handoff between assignment service and delivery service,
with comprehensive verification of intermediate states during delivery progression.

The enhancement focuses on verifying that entities progress through the expected
sequence of states during delivery, not just checking final outcomes.
"""

import pytest
import simpy
from unittest.mock import MagicMock

# Import core simulation components
from delivery_sim.entities.order import Order
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.pair import Pair
from delivery_sim.entities.delivery_unit import DeliveryUnit
from delivery_sim.entities.states import OrderState, DriverState, PairState, DeliveryUnitState
from delivery_sim.entities.restaurant import Restaurant
from delivery_sim.utils.entity_type_utils import EntityType
from delivery_sim.utils.location_utils import calculate_distance

# Import events
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.events.delivery_unit_events import DeliveryUnitAssignedEvent

# Import services
from delivery_sim.services.delivery_service import DeliveryService

# Import repositories
from delivery_sim.repositories.order_repository import OrderRepository
from delivery_sim.repositories.driver_repository import DriverRepository
from delivery_sim.repositories.delivery_unit_repository import DeliveryUnitRepository
from delivery_sim.repositories.pair_repository import PairRepository


# Import utilities
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.utils.logging_system import configure_logging, get_logger


class TestAssignmentDeliveryHandoffWithIntermediateVerification:
    """
    Test suite for assignment-delivery handoff with enhanced intermediate state verification.
    
    These tests verify not just that delivery processes complete successfully, but that
    entities progress through the expected sequence of intermediate states at the
    expected times during the delivery workflow.
    
    Key enhancement: Instead of running simulation until completion and checking only
    final states, these tests run simulation in phases and verify intermediate states
    at each phase, providing much more detailed verification of process correctness.
    """

    @pytest.fixture
    def test_environment(self):
        """Create a controlled test environment for delivery testing."""
        # Configure logging for test debugging
        configure_logging()
        
        # Create SimPy environment
        env = simpy.Environment()
        
        return env

    @pytest.fixture
    def test_config(self):
        """Create test configuration optimized for delivery testing."""
        # Create a mock config object that provides the attributes DeliveryService expects
        class TestConfig:
            def __init__(self):
                # DeliveryService expects driver_speed from config
                self.driver_speed = 1.0  # 1 km/min for predictable timing calculations
                
                # Other potential config values that services might need
                self.pairing_enabled = False  # Simplified for delivery testing
                
        return TestConfig()

    @pytest.fixture  
    def test_timing_config(self):
        """Create timing configuration for test calculations."""
        return {
            'service_time_buffer': 0.1  # Small buffer for service processing time
        }

    @pytest.fixture
    def test_repositories(self, test_environment):
        """Create repositories for test entity management."""
        return {
            'order': OrderRepository(),
            'driver': DriverRepository(), 
            'delivery_unit': DeliveryUnitRepository(),
            'pair': PairRepository()  # Add pair repository that DeliveryService expects
        }

    @pytest.fixture
    def delivery_service(self, test_environment, test_repositories, test_config):
        """Create delivery service for testing."""
        event_dispatcher = EventDispatcher()
        
        # Create DeliveryService with proper constructor signature
        service = DeliveryService(
            env=test_environment,
            event_dispatcher=event_dispatcher,
            driver_repository=test_repositories['driver'],
            order_repository=test_repositories['order'],
            pair_repository=test_repositories['pair'],
            delivery_unit_repository=test_repositories['delivery_unit'],
            config=test_config  # Pass config object, not individual parameters
        )
        
        return service, event_dispatcher

    def test_single_order_delivery_with_intermediate_verification(
        self, test_environment, test_repositories, delivery_service, test_config, test_timing_config
    ):
        """
        Test single order delivery process with comprehensive intermediate state verification.
        
        This enhanced test verifies that the delivery process follows the expected sequence:
        1. Initial state: Order ASSIGNED, Driver DELIVERING, DeliveryUnit IN_PROGRESS
        2. After reaching restaurant: Order PICKED_UP, Driver still DELIVERING
        3. After reaching customer: Order DELIVERED, Driver AVAILABLE, DeliveryUnit COMPLETED
        
        By checking intermediate states, we gain confidence that the delivery process
        follows the correct business logic sequence, not just that it eventually reaches
        the right final state.
        """
        # ARRANGE - Set up controlled delivery scenario
        env = test_environment
        service, event_dispatcher = delivery_service
        
        # Create restaurant for reference
        restaurant = Restaurant(
            restaurant_id="R1",
            location=[2.0, 2.0]
        )
        
        # Create test order in ASSIGNED state (ready for delivery)
        order = Order(
            order_id="O1", 
            restaurant_location=[2.0, 2.0],  # Restaurant location
            customer_location=[4.0, 4.0],   # Customer location (2.83 km from restaurant)
            arrival_time=0.0
        )
        order.state = OrderState.ASSIGNED  # Set to ASSIGNED (post-assignment state)
        order.assignment_time = env.now
        order.entity_type = EntityType.ORDER
        test_repositories['order'].add(order)
        
        # Create test driver in DELIVERING state (assigned to delivery)
        driver = Driver(
            driver_id="D1",
            initial_location=[0.0, 0.0],    # Starting position (2.83 km from restaurant)
            login_time=0.0,
            service_duration=60
        )
        driver.state = DriverState.DELIVERING  # Set to DELIVERING (post-assignment state)
        driver.entity_type = EntityType.DRIVER
        test_repositories['driver'].add(driver)
        
        # Create delivery unit linking order and driver
        # Note: DeliveryUnit automatically generates its ID based on the delivery_entity and driver
        delivery_unit = DeliveryUnit(
            delivery_entity=order,
            driver=driver,
            assignment_time=env.now
        )
        # The delivery unit will have automatically generated ID like "DU-O1-D1"
        test_repositories['delivery_unit'].add(delivery_unit)
        
        # Set assignment path after creation (this is typically done by AssignmentService)
        delivery_unit.assignment_path = "immediate"
        
        # Establish entity relationships (bidirectional references)
        order.delivery_unit = delivery_unit
        driver.current_delivery = delivery_unit
        
        # Verify initial state setup is correct
        assert order.state == OrderState.ASSIGNED, "Order should start in ASSIGNED state"
        assert driver.state == DriverState.DELIVERING, "Driver should start in DELIVERING state"
        assert delivery_unit.state == DeliveryUnitState.IN_PROGRESS, "DeliveryUnit should start IN_PROGRESS"
        
        # Calculate timing for phase-based verification
        # Phase 1: Driver travels to restaurant (distance = 2.83 km, speed = 1.0 km/min)
        restaurant_travel_time = calculate_distance(driver.location, order.restaurant_location) / test_config.driver_speed
        pickup_completion_time = restaurant_travel_time + test_timing_config['service_time_buffer']
        
        # Phase 2: Driver travels to customer (distance = 2.83 km, speed = 1.0 km/min)  
        customer_travel_time = calculate_distance(order.restaurant_location, order.customer_location) / test_config.driver_speed
        delivery_completion_time = pickup_completion_time + customer_travel_time + test_timing_config['service_time_buffer']
        
        # ACT & ASSERT - Phase 1: Trigger delivery and verify pickup phase
        
        # Dispatch delivery start event
        event_dispatcher.dispatch(DeliveryUnitAssignedEvent(
            timestamp=env.now,
            delivery_unit_id=delivery_unit.unit_id,  # Use the auto-generated ID
            entity_type=EntityType.ORDER,  # Specify entity type
            entity_id=order.order_id,  # Order ID
            driver_id=driver.driver_id  # Driver ID
        ))
        
        # Run simulation until pickup should be completed
        env.run(until=pickup_completion_time)
        
        # INTERMEDIATE VERIFICATION 1: Pickup phase completion
        # At this point, driver should have reached restaurant and picked up the order
        assert order.state == OrderState.PICKED_UP, \
            f"Order should be PICKED_UP after {pickup_completion_time:.2f} minutes"
        assert order.pickup_time is not None, "Order pickup time should be recorded"
        assert order.pickup_time <= env.now, "Pickup time should be <= current simulation time"
        assert order.pickup_time > order.assignment_time, "Pickup should occur after assignment"
        
        # Driver should still be in DELIVERING state (not yet completed)
        assert driver.state == DriverState.DELIVERING, \
            "Driver should still be DELIVERING during customer transit"
        
        # DeliveryUnit should still be IN_PROGRESS
        assert delivery_unit.state == DeliveryUnitState.IN_PROGRESS, \
            "DeliveryUnit should still be IN_PROGRESS during customer transit"
        
        # Driver location should be at restaurant (pickup location)
        expected_location = order.restaurant_location
        actual_location = driver.location
        location_distance = calculate_distance(expected_location, actual_location)
        assert location_distance < 0.1, \
            f"Driver should be at restaurant location after pickup (distance: {location_distance:.3f})"
        
        # ACT & ASSERT - Phase 2: Complete delivery and verify final states
        
        # Run simulation until delivery should be completed
        env.run(until=delivery_completion_time)
        
        # FINAL VERIFICATION: Delivery completion
        # Order should be fully delivered
        assert order.state == OrderState.DELIVERED, \
            f"Order should be DELIVERED after {delivery_completion_time:.2f} minutes"
        assert order.delivery_time is not None, "Order delivery time should be recorded"
        assert order.delivery_time <= env.now, "Delivery time should be <= current simulation time"
        assert order.delivery_time > order.pickup_time, "Delivery should occur after pickup"
        
        # Driver should be available again
        assert driver.state == DriverState.AVAILABLE, \
            "Driver should be AVAILABLE after completing delivery"
        assert driver.current_delivery is None, \
            "Driver should have no current delivery after completion"
        
        # DeliveryUnit should be completed
        assert delivery_unit.state == DeliveryUnitState.COMPLETED, \
            "DeliveryUnit should be COMPLETED after delivery"
        assert delivery_unit.completion_time is not None, \
            "DeliveryUnit completion time should be recorded"
        
        # Driver location should be at customer location (final delivery location)
        expected_final_location = order.customer_location
        actual_final_location = driver.location
        final_location_distance = calculate_distance(expected_final_location, actual_final_location)
        assert final_location_distance < 0.1, \
            f"Driver should be at customer location after delivery (distance: {final_location_distance:.3f})"
        
        # Verify complete timing sequence is logical
        assert order.assignment_time < order.pickup_time < order.delivery_time, \
            "Timing sequence should be: assignment < pickup < delivery"
        
        # Verify driver completed deliveries tracking
        assert len(driver.completed_deliveries) == 1, \
            "Driver should have one completed delivery"
        assert driver.completed_deliveries[0] is delivery_unit, \
            "Driver should reference the completed delivery unit"

    def test_pair_delivery_with_intermediate_verification(
        self, test_environment, test_repositories, delivery_service, test_config, test_timing_config
    ):
        """
        Test pair delivery process with comprehensive intermediate state verification.
        
        This enhanced test verifies the more complex paired delivery workflow:
        1. Initial state: Pair ASSIGNED, both Orders ASSIGNED, Driver DELIVERING
        2. After first pickup: First order PICKED_UP, pair still ASSIGNED
        3. After second pickup: Both orders PICKED_UP, pair still ASSIGNED  
        4. After first delivery: First order DELIVERED, second still PICKED_UP
        5. After second delivery: Both orders DELIVERED, pair COMPLETED
        
        The intermediate verification ensures that the sequential pickup and delivery
        logic works correctly for the more complex paired delivery scenario.
        """
        # ARRANGE - Set up controlled pair delivery scenario
        env = test_environment
        service, event_dispatcher = delivery_service
        
        # Create two orders that form a pair (same restaurant, nearby customers)
        order1 = Order(
            order_id="O1",
            restaurant_location=[3.0, 3.0],
            customer_location=[5.0, 5.0],  # 2.83 km from restaurant
            arrival_time=0.0
        )
        order1.state = OrderState.ASSIGNED
        order1.assignment_time = env.now
        order1.entity_type = EntityType.ORDER
        test_repositories['order'].add(order1)
        
        order2 = Order(
            order_id="O2", 
            restaurant_location=[3.0, 3.0],  # Same restaurant as order1
            customer_location=[6.0, 6.0],   # 4.24 km from restaurant, close to order1
            arrival_time=0.0
        )
        order2.state = OrderState.ASSIGNED
        order2.assignment_time = env.now
        order2.entity_type = EntityType.ORDER
        test_repositories['order'].add(order2)
        
        # Create pair containing both orders
        pair = Pair(order1=order1, order2=order2, creation_time=env.now)

        # Calculate and set the optimal sequence (simulating what pairing service does)
        # For same restaurant, optimal sequence is: restaurant → customer1 → customer2
        pair.optimal_sequence = [
            order1.restaurant_location,  # [3.0, 3.0] - pickup location
            order1.customer_location,    # [5.0, 5.0] - first delivery
            order2.customer_location     # [6.0, 6.0] - second delivery
        ]

        # Calculate the optimal cost (total distance of the sequence)
        restaurant_to_customer1 = calculate_distance(order1.restaurant_location, order1.customer_location)
        customer1_to_customer2 = calculate_distance(order1.customer_location, order2.customer_location)
        pair.optimal_cost = restaurant_to_customer1 + customer1_to_customer2

        # Now set the other attributes
        pair.state = PairState.ASSIGNED
        pair.assignment_time = env.now
        pair.entity_type = EntityType.PAIR
        test_repositories['pair'].add(pair)
        
        # Establish order-pair relationships
        order1.pair = pair
        order2.pair = pair
        
        # Create driver for pair delivery
        driver = Driver(
            driver_id="D1",
            initial_location=[1.0, 1.0],    # 2.83 km from restaurant
            login_time=0.0,
            service_duration=90
        )
        driver.state = DriverState.DELIVERING
        driver.entity_type = EntityType.DRIVER
        test_repositories['driver'].add(driver)
        
        # Create delivery unit for the pair
        # Note: DeliveryUnit automatically generates its ID based on the pair and driver
        delivery_unit = DeliveryUnit(
            delivery_entity=pair,
            driver=driver,
            assignment_time=env.now
        )
        # The delivery unit will have automatically generated ID like "DU-P-O1_O2-D1"
        test_repositories['delivery_unit'].add(delivery_unit)
        
        # Set assignment path after creation
        delivery_unit.assignment_path = "immediate"
        
        # Establish entity relationships
        pair.delivery_unit = delivery_unit
        driver.current_delivery = delivery_unit
        
        # Verify initial state setup
        assert order1.state == OrderState.ASSIGNED, "Order 1 should start ASSIGNED"
        assert order2.state == OrderState.ASSIGNED, "Order 2 should start ASSIGNED"
        assert pair.state == PairState.ASSIGNED, "Pair should start ASSIGNED"
        assert driver.state == DriverState.DELIVERING, "Driver should start DELIVERING"
        
        # Calculate timing for multi-phase verification
        # The delivery service should calculate optimal sequence for the pair
        # For this test, we'll assume order1 is delivered first, then order2
        
        # Phase 1: Travel to restaurant
        restaurant_travel_time = calculate_distance(driver.location, order1.restaurant_location) / test_config.driver_speed
        restaurant_arrival_time = restaurant_travel_time + test_timing_config['service_time_buffer']
        
        # Phase 2: Complete both pickups (since same restaurant)
        pickup_completion_time = restaurant_arrival_time + test_timing_config['service_time_buffer']
        
        # Phase 3: Deliver first order
        first_delivery_travel_time = calculate_distance(order1.restaurant_location, order1.customer_location) / test_config.driver_speed
        first_delivery_time = pickup_completion_time + first_delivery_travel_time + test_timing_config['service_time_buffer']
        
        # Phase 4: Deliver second order  
        second_delivery_travel_time = calculate_distance(order1.customer_location, order2.customer_location) / test_config.driver_speed
        second_delivery_time = first_delivery_time + second_delivery_travel_time + test_timing_config['service_time_buffer']
        
        # ACT & ASSERT - Execute delivery with intermediate verification
        
        # Dispatch delivery start event
        event_dispatcher.dispatch(DeliveryUnitAssignedEvent(
            timestamp=env.now,
            delivery_unit_id=delivery_unit.unit_id,  # Use the auto-generated ID
            entity_type=EntityType.PAIR,  # Specify entity type for pair
            entity_id=pair.pair_id,  # Pair ID (auto-generated)
            driver_id=driver.driver_id  # Driver ID
        ))
        
        # Phase 1: Verify pickup completion
        env.run(until=pickup_completion_time)
        
        # INTERMEDIATE VERIFICATION 1: Both orders should be picked up
        assert order1.state == OrderState.PICKED_UP, "Order 1 should be PICKED_UP after restaurant visit"
        assert order2.state == OrderState.PICKED_UP, "Order 2 should be PICKED_UP after restaurant visit"
        assert order1.pickup_time is not None, "Order 1 pickup time should be recorded"
        assert order2.pickup_time is not None, "Order 2 pickup time should be recorded"
        
        # Pair should still be assigned (not completed until all deliveries done)
        assert pair.state == PairState.ASSIGNED, "Pair should still be ASSIGNED during deliveries"
        
        # Driver should still be delivering
        assert driver.state == DriverState.DELIVERING, "Driver should still be DELIVERING"
        
        # Phase 2: Verify first delivery completion
        env.run(until=first_delivery_time)
        
        # INTERMEDIATE VERIFICATION 2: First order delivered, second still in transit
        assert order1.state == OrderState.DELIVERED, "Order 1 should be DELIVERED"
        assert order1.delivery_time is not None, "Order 1 delivery time should be recorded"
        
        # Second order should still be picked up (not yet delivered)
        assert order2.state == OrderState.PICKED_UP, "Order 2 should still be PICKED_UP"
        assert order2.delivery_time is None, "Order 2 delivery time should not be set yet"
        
        # Pair should still be in progress
        assert pair.state == PairState.ASSIGNED, "Pair should still be ASSIGNED with one order remaining"
        assert driver.state == DriverState.DELIVERING, "Driver should still be DELIVERING"
        
        # Phase 3: Complete second delivery
        env.run(until=second_delivery_time)
        
        # FINAL VERIFICATION: Complete pair delivery
        assert order1.state == OrderState.DELIVERED, "Order 1 should remain DELIVERED"
        assert order2.state == OrderState.DELIVERED, "Order 2 should be DELIVERED"
        assert order2.delivery_time is not None, "Order 2 delivery time should be recorded"
        
        # Pair should be completed
        assert pair.state == PairState.COMPLETED, "Pair should be COMPLETED after both deliveries"
        assert pair.completion_time is not None, "Pair completion time should be recorded"
        
        # Driver should be available
        assert driver.state == DriverState.AVAILABLE, "Driver should be AVAILABLE after pair completion"
        assert driver.current_delivery is None, "Driver should have no current delivery"
        
        # DeliveryUnit should be completed
        assert delivery_unit.state == DeliveryUnitState.COMPLETED, "DeliveryUnit should be COMPLETED"
        
        # Verify timing relationships for both orders
        assert order1.pickup_time < order1.delivery_time, "Order 1: pickup before delivery"
        assert order2.pickup_time < order2.delivery_time, "Order 2: pickup before delivery"
        assert order1.delivery_time < order2.delivery_time, "Order 1 delivered before Order 2"
        
        # Verify pair tracking
        assert len(pair.delivered_orders) == 2, "Pair should track both delivered orders"
        assert order1.order_id in pair.delivered_orders, "Pair should track order 1 delivery"
        assert order2.order_id in pair.delivered_orders, "Pair should track order 2 delivery"

    def test_delivery_timing_constraints_verification(
        self, test_environment, test_repositories, delivery_service, test_config, test_timing_config
    ):
        """
        Test that delivery process respects realistic timing constraints.
        
        This test specifically focuses on verifying that the timing relationships
        during delivery are logical and realistic, ensuring that the simulation
        produces believable delivery timelines.
        
        Key timing constraints verified:
        - Travel time matches distance/speed calculations
        - State transitions occur in logical sequence
        - No instantaneous actions (everything takes reasonable time)
        - Timing is recorded accurately for all major events
        """
        # ARRANGE - Set up scenario with known distances for timing verification
        env = test_environment
        service, event_dispatcher = delivery_service
        
        # Create order with precisely calculated distances
        order = Order(
            order_id="O1",
            restaurant_location=[0.0, 0.0],    # Origin point for easy calculation
            customer_location=[3.0, 4.0],      # Exactly 5.0 km from restaurant (3-4-5 triangle)
            arrival_time=0.0
        )
        order.state = OrderState.ASSIGNED
        order.assignment_time = env.now
        order.entity_type = EntityType.ORDER
        test_repositories['order'].add(order)
        
        # Create driver with known starting position
        driver = Driver(
            driver_id="D1", 
            initial_location=[0.0, 3.0],       # Exactly 3.0 km from restaurant
            login_time=0.0,
            service_duration=60
        )
        driver.state = DriverState.DELIVERING
        driver.entity_type = EntityType.DRIVER
        test_repositories['driver'].add(driver)
        
        # Create delivery unit
        # Note: DeliveryUnit automatically generates its ID based on the order and driver
        delivery_unit = DeliveryUnit(
            delivery_entity=order,
            driver=driver,
            assignment_time=env.now
        )
        # The delivery unit will have automatically generated ID like "DU-O1-D1"
        test_repositories['delivery_unit'].add(delivery_unit)
        
        # Set assignment path after creation
        delivery_unit.assignment_path = "immediate"
        
        # Establish relationships
        order.delivery_unit = delivery_unit
        driver.current_delivery = delivery_unit
        
        # Calculate expected timing with precision
        # Driver to restaurant: 3.0 km at 1.0 km/min = 3.0 minutes
        expected_restaurant_arrival = 3.0 + test_timing_config['service_time_buffer']
        
        # Restaurant to customer: 5.0 km at 1.0 km/min = 5.0 minutes  
        expected_delivery_completion = expected_restaurant_arrival + 5.0 + test_timing_config['service_time_buffer']
        
        # Record initial time for timing calculations
        start_time = env.now
        
        # ACT - Start delivery process
        event_dispatcher.dispatch(DeliveryUnitAssignedEvent(
            timestamp=env.now,
            delivery_unit_id=delivery_unit.unit_id,  # Use the auto-generated ID
            entity_type=EntityType.ORDER,  # Specify entity type
            entity_id=order.order_id,  # Order ID
            driver_id=driver.driver_id  # Driver ID
        ))
        
        # ASSERT - Verify timing precision at pickup
        env.run(until=expected_restaurant_arrival)
        
        # Verify pickup occurred and timing is accurate
        assert order.state == OrderState.PICKED_UP, "Order should be picked up at expected time"
        
        pickup_duration = order.pickup_time - start_time
        expected_pickup_duration = expected_restaurant_arrival
        timing_tolerance = 0.2  # Allow small tolerance for processing time
        
        assert abs(pickup_duration - expected_pickup_duration) < timing_tolerance, \
            f"Pickup should occur at {expected_pickup_duration:.2f} min, " \
            f"but occurred at {pickup_duration:.2f} min (difference: {abs(pickup_duration - expected_pickup_duration):.3f})"
        
        # ASSERT - Verify timing precision at delivery
        env.run(until=expected_delivery_completion)
        
        # Verify delivery occurred and timing is accurate
        assert order.state == OrderState.DELIVERED, "Order should be delivered at expected time"
        
        delivery_duration = order.delivery_time - start_time
        expected_delivery_duration = expected_delivery_completion
        
        assert abs(delivery_duration - expected_delivery_duration) < timing_tolerance, \
            f"Delivery should occur at {expected_delivery_duration:.2f} min, " \
            f"but occurred at {delivery_duration:.2f} min (difference: {abs(delivery_duration - expected_delivery_duration):.3f})"
        
        # Verify that timing sequence is strictly increasing
        assert start_time < order.pickup_time < order.delivery_time, \
            "Timing sequence must be strictly increasing: start < pickup < delivery"
        
        # Verify reasonable timing intervals
        pickup_interval = order.pickup_time - start_time
        delivery_interval = order.delivery_time - order.pickup_time
        
        # Pickup interval should be roughly the travel time to restaurant
        assert 2.5 < pickup_interval < 4.0, \
            f"Pickup interval should be ~3 minutes (travel time), got {pickup_interval:.2f}"
        
        # Delivery interval should be roughly the travel time to customer
        assert 4.5 < delivery_interval < 6.0, \
            f"Delivery interval should be ~5 minutes (travel time), got {delivery_interval:.2f}"
        
        # Verify driver location matches expected final position
        expected_final_location = order.customer_location
        actual_final_location = driver.location
        location_precision = calculate_distance(expected_final_location, actual_final_location)
        
        assert location_precision < 0.1, \
            f"Driver should be at customer location (precision: {location_precision:.3f})"

