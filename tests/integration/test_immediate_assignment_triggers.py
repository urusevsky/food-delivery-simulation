# tests/integration/test_immediate_assignment_triggers.py
"""
Integration tests for Area 2: Immediate Assignment Triggering Mechanisms

This test file focuses on verifying that various events correctly trigger
immediate assignment attempts in the AssignmentService. We're testing the
"entry points" into assignment logic, not the outcomes of those attempts.

Key principle: These tests verify that events cause assignment attempts,
but don't test whether those attempts succeed or fail (that's Area 3's job).
"""

import pytest
import simpy
from unittest.mock import Mock

# Import core components
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.events.order_events import OrderCreatedEvent
from delivery_sim.events.pair_events import PairCreatedEvent, PairingFailedEvent
from delivery_sim.events.driver_events import DriverLoggedInEvent, DriverAvailableForAssignmentEvent
from delivery_sim.repositories.order_repository import OrderRepository
from delivery_sim.repositories.driver_repository import DriverRepository
from delivery_sim.repositories.pair_repository import PairRepository
from delivery_sim.repositories.delivery_unit_repository import DeliveryUnitRepository
from delivery_sim.services.assignment_service import AssignmentService
from delivery_sim.entities.order import Order
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.pair import Pair
from delivery_sim.entities.states import OrderState, DriverState
from delivery_sim.utils.entity_type_utils import EntityType


class TestImmediateAssignmentTriggers:
    """
    Test suite verifying that all five pathways correctly trigger immediate assignment attempts.
    
    The five triggers we're testing:
    1. OrderCreatedEvent (when pairing is disabled)
    2. PairCreatedEvent (when pairing is enabled)
    3. PairingFailedEvent (when pairing is enabled)
    4. DriverLoggedInEvent (new driver enters system)
    5. DriverAvailableForAssignmentEvent (driver completes delivery)
    """
    
    @pytest.fixture
    def test_config_pairing_disabled(self):
        """Configuration with pairing disabled - orders go directly to assignment."""
        class TestConfig:
            def __init__(self):
                # Assignment parameters
                self.immediate_assignment_threshold = 5.0
                self.periodic_interval = 10.0
                self.throughput_factor = 1.0
                self.age_factor = 0.1
                self.driver_speed = 0.5
                
                # Key setting: pairing is disabled
                self.pairing_enabled = False
        
        return TestConfig()
    
    @pytest.fixture
    def test_config_pairing_enabled(self):
        """Configuration with pairing enabled - orders first attempt pairing."""
        class TestConfig:
            def __init__(self):
                # Assignment parameters (same as above)
                self.immediate_assignment_threshold = 5.0
                self.periodic_interval = 10.0
                self.throughput_factor = 1.0
                self.age_factor = 0.1
                self.driver_speed = 0.5
                
                # Key setting: pairing is enabled
                self.pairing_enabled = True
        
        return TestConfig()
    
    @pytest.fixture
    def test_environment(self):
        """Set up the common test environment with all necessary components."""
        env = simpy.Environment()
        event_dispatcher = EventDispatcher()
        order_repo = OrderRepository()
        driver_repo = DriverRepository()
        pair_repo = PairRepository()
        delivery_unit_repo = DeliveryUnitRepository()
        
        return {
            "env": env,
            "event_dispatcher": event_dispatcher,
            "order_repo": order_repo,
            "driver_repo": driver_repo,
            "pair_repo": pair_repo,
            "delivery_unit_repo": delivery_unit_repo
        }
    
    # ===== Trigger Tests from Delivery Entity Perspective =====
    
    def test_order_created_triggers_assignment_when_pairing_disabled(self, test_environment, test_config_pairing_disabled):
        """
        Test Trigger #1: OrderCreatedEvent → immediate assignment attempt (pairing disabled).
        
        When pairing is disabled, new orders should go directly to assignment evaluation.
        This represents the simplest path where orders bypass pairing entirely.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_disabled  # Pairing is disabled
        
        # Create the assignment service with our configuration
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=config
        )
        
        # Create a spy to track assignment attempts from delivery entity perspective
        assignment_attempts = []
        original_method = assignment_service.attempt_immediate_assignment_from_delivery_entity
        
        def spy_assignment_attempt(entity):
            # Record what entity triggered the attempt
            entity_id = entity.order_id if hasattr(entity, 'order_id') else entity.pair_id
            assignment_attempts.append({
                'entity_type': entity.entity_type,
                'entity_id': entity_id
            })
            # Call the original method to maintain normal behavior
            return original_method(entity)
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = spy_assignment_attempt
        
        # Create a test order
        test_order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[7, 7],
            arrival_time=env.now
        )
        order_repo.add(test_order)
        
        # ACT - Dispatch OrderCreatedEvent
        event_dispatcher.dispatch(OrderCreatedEvent(
            timestamp=env.now,
            order_id=test_order.order_id,
            restaurant_id="R1",
            restaurant_location=test_order.restaurant_location,
            customer_location=test_order.customer_location
        ))
        
        # Run briefly to allow event processing
        env.run(until=0.1)
        
        # ASSERT - Verify the assignment attempt was triggered
        assert len(assignment_attempts) == 1, "Should trigger exactly one assignment attempt"
        
        attempt = assignment_attempts[0]
        assert attempt['entity_type'] == EntityType.ORDER, "Should attempt assignment for an order"
        assert attempt['entity_id'] == test_order.order_id, "Should attempt assignment for the correct order"
    
    def test_pair_created_triggers_assignment_when_pairing_enabled(self, test_environment, test_config_pairing_enabled):
        """
        Test Trigger #2: PairCreatedEvent → immediate assignment attempt.
        
        When pairing succeeds, the newly created pair should trigger an assignment attempt.
        This represents the path where orders are successfully combined before assignment.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_enabled  # Pairing is enabled
        
        # Create the assignment service
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=config
        )
        
        # Create a spy for assignment attempts
        assignment_attempts = []
        original_method = assignment_service.attempt_immediate_assignment_from_delivery_entity
        
        def spy_assignment_attempt(entity):
            entity_id = entity.order_id if hasattr(entity, 'order_id') else entity.pair_id
            assignment_attempts.append({
                'entity_type': entity.entity_type,
                'entity_id': entity_id
            })
            return original_method(entity)
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = spy_assignment_attempt
        
        # Create a test pair (representing successful pairing)
        order1 = Order("O1", [3, 3], [7, 7], env.now)
        order2 = Order("O2", [3, 3], [8, 8], env.now)
        
        test_pair = Pair(order1, order2, env.now)
        test_pair.optimal_sequence = [[3, 3], [7, 7], [8, 8]]
        test_pair.optimal_cost = 5.0
        pair_repo.add(test_pair)
        
        # ACT - Dispatch PairCreatedEvent
        event_dispatcher.dispatch(PairCreatedEvent(
            timestamp=env.now,
            pair_id=test_pair.pair_id,
            order1_id=order1.order_id,
            order2_id=order2.order_id
        ))
        
        # Run briefly to allow event processing
        env.run(until=0.1)
        
        # ASSERT - Verify the assignment attempt was triggered
        assert len(assignment_attempts) == 1, "Should trigger exactly one assignment attempt"
        
        attempt = assignment_attempts[0]
        assert attempt['entity_type'] == EntityType.PAIR, "Should attempt assignment for a pair"
        assert attempt['entity_id'] == test_pair.pair_id, "Should attempt assignment for the correct pair"
    
    def test_pairing_failed_triggers_assignment_when_pairing_enabled(self, test_environment, test_config_pairing_enabled):
        """
        Test Trigger #3: PairingFailedEvent → immediate assignment attempt.
        
        When pairing fails (no compatible partner found), the unpaired order should
        still trigger an assignment attempt as a single order. This is the fallback path.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_enabled  # Pairing is enabled
        
        # Create the assignment service
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=config
        )
        
        # Create a spy for assignment attempts
        assignment_attempts = []
        original_method = assignment_service.attempt_immediate_assignment_from_delivery_entity
        
        def spy_assignment_attempt(entity):
            entity_id = entity.order_id if hasattr(entity, 'order_id') else entity.pair_id
            assignment_attempts.append({
                'entity_type': entity.entity_type,
                'entity_id': entity_id
            })
            return original_method(entity)
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = spy_assignment_attempt
        
        # Create a test order that failed to find a pair
        unpaired_order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[7, 7],
            arrival_time=env.now
        )
        unpaired_order.entity_type = EntityType.ORDER
        order_repo.add(unpaired_order)
        
        # ACT - Dispatch PairingFailedEvent
        event_dispatcher.dispatch(PairingFailedEvent(
            timestamp=env.now,
            order_id=unpaired_order.order_id
        ))
        
        # Run briefly to allow event processing
        env.run(until=0.1)
        
        # ASSERT - Verify the assignment attempt was triggered
        assert len(assignment_attempts) == 1, "Should trigger exactly one assignment attempt"
        
        attempt = assignment_attempts[0]
        assert attempt['entity_type'] == EntityType.ORDER, "Should attempt assignment for a single order"
        assert attempt['entity_id'] == unpaired_order.order_id, "Should attempt assignment for the correct order"
    
    # ===== Trigger Tests from Driver Perspective =====
    
    def test_driver_login_triggers_assignment_attempt(self, test_environment, test_config_pairing_disabled):
        """
        Test Trigger #4: DriverLoggedInEvent → immediate assignment attempt.
        
        When a new driver logs into the system, they should immediately check if
        there are any waiting delivery entities they can serve.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_disabled
        
        # Create the assignment service
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=config
        )
        
        # Create a spy for assignment attempts from driver perspective
        assignment_attempts = []
        original_method = assignment_service.attempt_immediate_assignment_from_driver
        
        def spy_assignment_attempt(driver):
            assignment_attempts.append({
                'driver_id': driver.driver_id,
                'trigger': 'driver_login'
            })
            return original_method(driver)
        
        assignment_service.attempt_immediate_assignment_from_driver = spy_assignment_attempt
        
        # Create a test driver
        test_driver = Driver(
            driver_id="D1",
            initial_location=[5, 5],
            login_time=env.now,
            service_duration=120  # 2 hours
        )
        driver_repo.add(test_driver)
        
        # ACT - Dispatch DriverLoggedInEvent
        event_dispatcher.dispatch(DriverLoggedInEvent(
            timestamp=env.now,
            driver_id=test_driver.driver_id,
            initial_location=test_driver.location,
            service_duration=test_driver.service_duration
        ))
        
        # Run briefly to allow event processing
        env.run(until=0.1)
        
        # ASSERT - Verify the assignment attempt was triggered
        assert len(assignment_attempts) == 1, "Should trigger exactly one assignment attempt"
        
        attempt = assignment_attempts[0]
        assert attempt['driver_id'] == test_driver.driver_id, "Should attempt assignment for the correct driver"
        assert attempt['trigger'] == 'driver_login', "Should be triggered by driver login"
    
    def test_driver_available_for_assignment_triggers_assignment_attempt(self, test_environment, test_config_pairing_disabled):
        """
        Test Trigger #5: DriverAvailableForAssignmentEvent → immediate assignment attempt.
        
        When a driver completes a delivery and is still eligible to work (not overdue),
        they should check for new delivery opportunities. This event is dispatched by
        DriverSchedulingService after verifying the driver isn't overdue for logout.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_disabled
        
        # Create the assignment service
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=config
        )
        
        # Create a spy for assignment attempts from driver perspective
        assignment_attempts = []
        original_method = assignment_service.attempt_immediate_assignment_from_driver
        
        def spy_assignment_attempt(driver):
            assignment_attempts.append({
                'driver_id': driver.driver_id,
                'trigger': 'driver_available'
            })
            return original_method(driver)
        
        assignment_service.attempt_immediate_assignment_from_driver = spy_assignment_attempt
        
        # Create a test driver who just completed a delivery
        test_driver = Driver(
            driver_id="D1",
            initial_location=[5, 5],
            login_time=env.now - 30,  # Logged in 30 minutes ago
            service_duration=120      # Plans to work 120 minutes total
        )
        test_driver.state = DriverState.AVAILABLE  # Just finished delivery
        driver_repo.add(test_driver)
        
        # ACT - Dispatch DriverAvailableForAssignmentEvent
        # Note: This event is normally dispatched by DriverSchedulingService
        # after verifying the driver isn't overdue for logout
        event_dispatcher.dispatch(DriverAvailableForAssignmentEvent(
            timestamp=env.now,
            driver_id=test_driver.driver_id
        ))
        
        # Run briefly to allow event processing
        env.run(until=0.1)
        
        # ASSERT - Verify the assignment attempt was triggered
        assert len(assignment_attempts) == 1, "Should trigger exactly one assignment attempt"
        
        attempt = assignment_attempts[0]
        assert attempt['driver_id'] == test_driver.driver_id, "Should attempt assignment for the correct driver"
        assert attempt['trigger'] == 'driver_available', "Should be triggered by driver availability"
    
    # ===== Verification Tests =====
    
    def test_no_duplicate_triggers_for_same_event(self, test_environment, test_config_pairing_disabled):
        """
        Verify that a single event doesn't trigger multiple assignment attempts.
        
        This is a safeguard test to ensure our event handling doesn't have
        duplicate registrations or circular triggering patterns.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_disabled
        
        # Create the assignment service
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=config
        )
        
        # Create spies for both types of assignment attempts
        entity_attempts = []
        driver_attempts = []
        
        original_entity_method = assignment_service.attempt_immediate_assignment_from_delivery_entity
        original_driver_method = assignment_service.attempt_immediate_assignment_from_driver
        
        def spy_entity_attempt(entity):
            entity_attempts.append(entity)
            return original_entity_method(entity)
        
        def spy_driver_attempt(driver):
            driver_attempts.append(driver)
            return original_driver_method(driver)
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = spy_entity_attempt
        assignment_service.attempt_immediate_assignment_from_driver = spy_driver_attempt
        
        # Create a test order
        test_order = Order("O1", [3, 3], [7, 7], env.now)
        order_repo.add(test_order)
        
        # ACT - Dispatch a single OrderCreatedEvent
        event_dispatcher.dispatch(OrderCreatedEvent(
            timestamp=env.now,
            order_id=test_order.order_id,
            restaurant_id="R1",
            restaurant_location=test_order.restaurant_location,
            customer_location=test_order.customer_location
        ))
        
        # Run with extra time to ensure no delayed duplicate triggers
        env.run(until=1.0)
        
        # ASSERT - Verify only one attempt was triggered
        assert len(entity_attempts) == 1, "Should trigger exactly one entity-based attempt"
        assert len(driver_attempts) == 0, "Should not trigger any driver-based attempts"
    
    def test_configuration_determines_order_event_routing(self, test_environment, test_config_pairing_disabled, test_config_pairing_enabled):
        """
        Verify that pairing configuration correctly determines whether OrderCreatedEvent
        triggers immediate assignment or not.
        
        This test confirms that the configuration-based routing works as expected.
        """
        # ARRANGE - First test with pairing disabled
        env = test_environment["env"]
        event_dispatcher = EventDispatcher()  # Fresh dispatcher for each config
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        
        # Test 1: Pairing disabled - should trigger assignment
        assignment_service_no_pairing = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=test_config_pairing_disabled
        )
        
        attempts_no_pairing = []
        original_method = assignment_service_no_pairing.attempt_immediate_assignment_from_delivery_entity
        assignment_service_no_pairing.attempt_immediate_assignment_from_delivery_entity = \
            lambda e: attempts_no_pairing.append(e) or original_method(e)
        
        # Create and dispatch order event
        test_order1 = Order("O1", [3, 3], [7, 7], env.now)
        order_repo.add(test_order1)
        
        event_dispatcher.dispatch(OrderCreatedEvent(
            timestamp=env.now,
            order_id=test_order1.order_id,
            restaurant_id="R1",
            restaurant_location=test_order1.restaurant_location,
            customer_location=test_order1.customer_location
        ))
        
        env.run(until=0.1)
        
        # Test 2: Pairing enabled - should NOT trigger assignment
        event_dispatcher2 = EventDispatcher()  # Fresh dispatcher
        assignment_service_with_pairing = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher2,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=test_config_pairing_enabled
        )
        
        attempts_with_pairing = []
        original_method2 = assignment_service_with_pairing.attempt_immediate_assignment_from_delivery_entity
        assignment_service_with_pairing.attempt_immediate_assignment_from_delivery_entity = \
            lambda e: attempts_with_pairing.append(e) or original_method2(e)
        
        # Create and dispatch another order event
        test_order2 = Order("O2", [4, 4], [8, 8], env.now)
        order_repo.add(test_order2)
        
        event_dispatcher2.dispatch(OrderCreatedEvent(
            timestamp=env.now,
            order_id=test_order2.order_id,
            restaurant_id="R2",
            restaurant_location=test_order2.restaurant_location,
            customer_location=test_order2.customer_location
        ))
        
        env.run(until=0.2)
        
        # ASSERT
        assert len(attempts_no_pairing) == 1, "With pairing disabled, OrderCreatedEvent should trigger assignment"
        assert len(attempts_with_pairing) == 0, "With pairing enabled, OrderCreatedEvent should NOT trigger assignment"