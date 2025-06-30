# tests/integration/test_immediate_assignment_triggers.py
"""
Integration tests for Area 2: Immediate Assignment Triggering Mechanisms (Priority Scoring)

This test file focuses on verifying that various events correctly trigger
immediate assignment attempts in the AssignmentService. We're testing the
"entry points" into assignment logic, not the outcomes of those attempts.

Key principle: These tests verify that events cause assignment attempts,
but don't test whether those attempts succeed or fail (that's Area 3's job).

Updated for priority scoring system:
- AssignmentService now requires priority_scorer parameter
- Tests focus on trigger verification, not assignment outcomes
- Uses mock priority scorer to isolate trigger testing from scoring logic
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
    
    Updated for priority scoring system - focuses on trigger mechanisms.
    """
    
    @pytest.fixture
    def test_config_pairing_disabled(self):
        """Configuration with pairing disabled - orders go directly to assignment."""
        class TestConfig:
            def __init__(self):
                # Assignment parameters
                self.immediate_assignment_threshold = 75.0  # Priority score threshold
                self.periodic_interval = 10.0
                self.driver_speed = 0.5
                
                # Key setting: pairing is disabled
                self.pairing_enabled = False
        
        return TestConfig()
    
    @pytest.fixture
    def test_config_pairing_enabled(self):
        """Configuration with pairing enabled - orders first attempt pairing."""
        class TestConfig:
            def __init__(self):
                # Assignment parameters
                self.immediate_assignment_threshold = 75.0  # Priority score threshold
                self.periodic_interval = 10.0
                self.driver_speed = 0.5
                
                # Key setting: pairing is enabled
                self.pairing_enabled = True
        
        return TestConfig()
    
    @pytest.fixture
    def test_environment(self):
        """Set up the test environment with all repositories and event dispatcher."""
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
    
    @pytest.fixture
    def mock_priority_scorer(self):
        """Create a mock priority scorer that provides consistent results."""
        mock_scorer = Mock()
        # Use return_value with correct component keys to avoid StopIteration
        mock_scorer.calculate_priority_score.return_value = (80.0, {
            'distance_score': 0.8,        # Fixed: was 'distance_score' 
            'throughput_score': 0.0,      # Fixed: was 'time_score'
            'fairness_score': 0.9,        # Fixed: was 'efficiency_score'
            'combined_score_0_1': 0.80,
            'total_distance': 8.5,
            'num_orders': 1,
            'assignment_delay_minutes': 5.0
        })
        return mock_scorer
    
    # ===== Trigger 1: OrderCreatedEvent (Pairing Disabled) =====
    
    def test_order_created_triggers_immediate_assignment_when_pairing_disabled(
        self, test_environment, test_config_pairing_disabled, mock_priority_scorer
    ):
        """
        Test that OrderCreatedEvent triggers immediate assignment when pairing is disabled.
        
        When pairing is disabled, orders should go directly to assignment rather than
        attempting to pair with other orders first.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_disabled
        
        # Create assignment service
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            priority_scorer=mock_priority_scorer,
            config=config
        )
        
        # Create an order in the repository
        test_order = Order("O1", [3, 3], [5, 5], env.now)
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        # Create a driver to make assignment possible
        test_driver = Driver("D1", [2, 2], env.now, 120)
        test_driver.entity_type = EntityType.DRIVER
        test_driver.state = DriverState.AVAILABLE
        driver_repo.add(test_driver)
        
        # Mock the assignment attempt method to verify it gets called
        original_attempt = assignment_service.attempt_immediate_assignment_from_delivery_entity
        assignment_attempts = []
        
        def track_assignment_attempts(entity):
            assignment_attempts.append(entity)
            return original_attempt(entity)
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = track_assignment_attempts
        
        # ACT - Dispatch OrderCreatedEvent with correct signature
        event = OrderCreatedEvent(
            timestamp=env.now,
            order_id="O1",
            restaurant_id="R1",  # Added required parameter
            restaurant_location=[3, 3],  # Added required parameter
            customer_location=[5, 5]  # Added required parameter
        )
        event_dispatcher.dispatch(event)
        
        # Allow event processing
        env.run(until=0.1)
        
        # ASSERT
        # Verify that assignment attempt was triggered
        assert len(assignment_attempts) == 1, "OrderCreatedEvent should trigger assignment attempt"
        assert assignment_attempts[0] is test_order, "Assignment attempt should be for the created order"
        
        # Verify priority scorer was used (indicates full assignment flow)
        mock_priority_scorer.calculate_priority_score.assert_called()
    
    # ===== Trigger 2: PairCreatedEvent (Pairing Enabled) =====
    
    def test_pair_created_triggers_immediate_assignment_when_pairing_enabled(
        self, test_environment, test_config_pairing_enabled, mock_priority_scorer
    ):
        """
        Test that PairCreatedEvent triggers immediate assignment when pairing is enabled.
        
        When pairing is enabled, pairs should trigger assignment attempts when they are formed.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_enabled
        
        # Create assignment service
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            priority_scorer=mock_priority_scorer,
            config=config
        )
        
        # Create orders and pair
        order1 = Order("O1", [3, 3], [5, 5], env.now)
        order2 = Order("O2", [3, 3], [6, 6], env.now)
        test_pair = Pair(order1, order2, env.now)
        
        order_repo.add(order1)
        order_repo.add(order2)
        pair_repo.add(test_pair)
        
        # Create a driver to make assignment possible
        test_driver = Driver("D1", [2, 2], env.now, 120)
        test_driver.entity_type = EntityType.DRIVER
        test_driver.state = DriverState.AVAILABLE
        driver_repo.add(test_driver)
        
        # Mock the assignment attempt method
        original_attempt = assignment_service.attempt_immediate_assignment_from_delivery_entity
        assignment_attempts = []
        
        def track_assignment_attempts(entity):
            assignment_attempts.append(entity)
            return original_attempt(entity)
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = track_assignment_attempts
        
        # ACT - Dispatch PairCreatedEvent with correct signature
        event = PairCreatedEvent(
            timestamp=env.now,
            pair_id=test_pair.pair_id,
            order1_id="O1",  # Added required parameter
            order2_id="O2"   # Added required parameter
        )
        event_dispatcher.dispatch(event)
        
        # Allow event processing
        env.run(until=0.1)
        
        # ASSERT
        # Verify that assignment attempt was triggered for the pair
        assert len(assignment_attempts) == 1, "PairCreatedEvent should trigger assignment attempt"
        assert assignment_attempts[0] is test_pair, "Assignment attempt should be for the created pair"
        
        # Verify priority scorer was used
        mock_priority_scorer.calculate_priority_score.assert_called()
    
    # ===== Trigger 3: PairingFailedEvent (Pairing Enabled) =====
    
    def test_pairing_failed_triggers_immediate_assignment_when_pairing_enabled(
        self, test_environment, test_config_pairing_enabled, mock_priority_scorer
    ):
        """
        Test that PairingFailedEvent triggers immediate assignment when pairing is enabled.
        
        When an order fails to pair, it should be assigned as a single order.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_enabled
        
        # Create assignment service
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            priority_scorer=mock_priority_scorer,
            config=config
        )
        
        # Create an order that failed pairing
        test_order = Order("O1", [3, 3], [5, 5], env.now)
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        # Create a driver to make assignment possible
        test_driver = Driver("D1", [2, 2], env.now, 120)
        test_driver.entity_type = EntityType.DRIVER
        test_driver.state = DriverState.AVAILABLE
        driver_repo.add(test_driver)
        
        # Mock the assignment attempt method
        original_attempt = assignment_service.attempt_immediate_assignment_from_delivery_entity
        assignment_attempts = []
        
        def track_assignment_attempts(entity):
            assignment_attempts.append(entity)
            return original_attempt(entity)
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = track_assignment_attempts
        
        # ACT - Dispatch PairingFailedEvent
        event = PairingFailedEvent(timestamp=env.now, order_id="O1")
        event_dispatcher.dispatch(event)
        
        # Allow event processing
        env.run(until=0.1)
        
        # ASSERT
        # Verify that assignment attempt was triggered
        assert len(assignment_attempts) == 1, "PairingFailedEvent should trigger assignment attempt"
        assert assignment_attempts[0] is test_order, "Assignment attempt should be for the failed order"
        
        # Verify priority scorer was used
        mock_priority_scorer.calculate_priority_score.assert_called()
    
    # ===== Trigger 4: DriverLoggedInEvent =====
    
    def test_driver_login_triggers_immediate_assignment(
        self, test_environment, test_config_pairing_disabled, mock_priority_scorer
    ):
        """
        Test that DriverLoggedInEvent triggers immediate assignment attempts.
        
        When a new driver becomes available, the system should attempt to assign
        pending orders to that driver.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_disabled
        
        # Create assignment service
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            priority_scorer=mock_priority_scorer,
            config=config
        )
        
        # Create pending orders
        test_order = Order("O1", [3, 3], [5, 5], env.now)
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        # Create a driver (will be added to repo by the login event handling)
        test_driver = Driver("D1", [2, 2], env.now, 120)
        test_driver.entity_type = EntityType.DRIVER
        test_driver.state = DriverState.AVAILABLE
        driver_repo.add(test_driver)
        
        # Mock the assignment attempt method
        original_attempt = assignment_service.attempt_immediate_assignment_from_driver
        assignment_attempts = []
        
        def track_assignment_attempts(driver):
            assignment_attempts.append(driver)
            return original_attempt(driver)
        
        assignment_service.attempt_immediate_assignment_from_driver = track_assignment_attempts
        
        # ACT - Dispatch DriverLoggedInEvent with correct signature
        event = DriverLoggedInEvent(
            timestamp=env.now,
            driver_id="D1",
            initial_location=[2, 2],  # Added required parameter
            service_duration=120      # Added required parameter
        )
        event_dispatcher.dispatch(event)
        
        # Allow event processing
        env.run(until=0.1)
        
        # ASSERT
        # Verify that assignment attempt was triggered
        assert len(assignment_attempts) == 1, "DriverLoggedInEvent should trigger assignment attempt"
        assert assignment_attempts[0] is test_driver, "Assignment attempt should be for the logged-in driver"
        
        # Verify priority scorer was used
        mock_priority_scorer.calculate_priority_score.assert_called()
    
    # ===== Trigger 5: DriverAvailableForAssignmentEvent =====
    
    def test_driver_available_triggers_immediate_assignment(
        self, test_environment, test_config_pairing_disabled, mock_priority_scorer
    ):
        """
        Test that DriverAvailableForAssignmentEvent triggers immediate assignment attempts.
        
        When a driver becomes available after completing a delivery, the system
        should attempt to assign new orders to that driver.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_disabled
        
        # Create assignment service
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            priority_scorer=mock_priority_scorer,
            config=config
        )
        
        # Create pending orders
        test_order = Order("O1", [3, 3], [5, 5], env.now)
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        # Create an available driver
        test_driver = Driver("D1", [2, 2], env.now, 120)
        test_driver.entity_type = EntityType.DRIVER
        test_driver.state = DriverState.AVAILABLE
        driver_repo.add(test_driver)
        
        # Mock the assignment attempt method
        original_attempt = assignment_service.attempt_immediate_assignment_from_driver
        assignment_attempts = []
        
        def track_assignment_attempts(driver):
            assignment_attempts.append(driver)
            return original_attempt(driver)
        
        assignment_service.attempt_immediate_assignment_from_driver = track_assignment_attempts
        
        # ACT - Dispatch DriverAvailableForAssignmentEvent
        event = DriverAvailableForAssignmentEvent(timestamp=env.now, driver_id="D1")
        event_dispatcher.dispatch(event)
        
        # Allow event processing
        env.run(until=0.1)
        
        # ASSERT
        # Verify that assignment attempt was triggered
        assert len(assignment_attempts) == 1, "DriverAvailableForAssignmentEvent should trigger assignment attempt"
        assert assignment_attempts[0] is test_driver, "Assignment attempt should be for the available driver"
        
        # Verify priority scorer was used
        mock_priority_scorer.calculate_priority_score.assert_called()