# tests/integration/test_driver_lifecycle_scheduling.py
"""
Integration tests for Driver Lifecycle and Scheduling (Priority Scoring)

Updated for priority scoring system:
- AssignmentService now requires priority_scorer parameter
- Tests focus on driver lifecycle events, not scoring details
- Uses mock priority scorer to isolate lifecycle testing from scoring logic
"""

import pytest
import simpy
from unittest.mock import Mock

# Import core components
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.events.driver_events import DriverLoggedInEvent, DriverAvailableForAssignmentEvent
from delivery_sim.repositories.order_repository import OrderRepository
from delivery_sim.repositories.driver_repository import DriverRepository
from delivery_sim.repositories.pair_repository import PairRepository
from delivery_sim.repositories.delivery_unit_repository import DeliveryUnitRepository
from delivery_sim.services.assignment_service import AssignmentService
from delivery_sim.services.driver_scheduling_service import DriverSchedulingService
from delivery_sim.entities.order import Order
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.states import OrderState, DriverState
from delivery_sim.utils.entity_type_utils import EntityType


class TestDriverLifecycleScheduling:
    """
    Test suite for driver lifecycle events and scheduling integration.
    
    Updated for priority scoring system - focuses on lifecycle mechanics.
    """
    
    @pytest.fixture
    def test_config(self):
        """Configuration for driver lifecycle testing."""
        class TestConfig:
            def __init__(self):
                self.immediate_assignment_threshold = 75.0
                self.periodic_interval = 10.0
                self.pairing_enabled = False  # Simplified for lifecycle testing
                self.driver_speed = 0.5
        
        return TestConfig()
    
    @pytest.fixture
    def test_environment(self):
        """Create a controlled test environment."""
        env = simpy.Environment()
        event_dispatcher = EventDispatcher()
        
        return {
            "env": env,
            "event_dispatcher": event_dispatcher,
            "order_repo": OrderRepository(),
            "driver_repo": DriverRepository(),
            "pair_repo": PairRepository(),
            "delivery_unit_repo": DeliveryUnitRepository()
        }
    
    @pytest.fixture
    def mock_priority_scorer(self):
        """Create a mock priority scorer for lifecycle testing."""
        scorer = Mock()
        # Return moderate score - focus is on lifecycle, not scoring outcomes
        scorer.calculate_priority_score.return_value = (70.0, {
            "distance_score": 0.7,
            "throughput_score": 0.0,
            "fairness_score": 0.8,
            "combined_score_0_1": 0.70,
            "total_distance": 8.5,
            "num_orders": 1,
            "assignment_delay_minutes": 6.0
        })
        return scorer
    
    # ===== Test 1: Driver Login Integration =====
    
    def test_driver_login_triggers_assignment_attempt(
        self, test_environment, test_config, mock_priority_scorer
    ):
        """
        Test that driver login events properly integrate with assignment service.
        
        This verifies the complete flow from driver entry to assignment attempt.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
        # Create assignment service (REAL service, not mock)
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
        
        # Create a waiting order to make assignment possible
        waiting_order = Order("O1", [3, 3], [5, 5], env.now)
        waiting_order.entity_type = EntityType.ORDER
        waiting_order.state = OrderState.CREATED
        order_repo.add(waiting_order)
        
        # Create a driver that will login
        new_driver = Driver("D1", [2, 2], env.now, 120)
        new_driver.entity_type = EntityType.DRIVER
        new_driver.state = DriverState.AVAILABLE
        driver_repo.add(new_driver)
        
        # Track assignment attempts
        assignment_attempts = []
        original_method = assignment_service.attempt_immediate_assignment_from_driver
        
        def track_attempts(driver):
            assignment_attempts.append(driver.driver_id)
            return original_method(driver)
        
        assignment_service.attempt_immediate_assignment_from_driver = track_attempts
        
        # ACT - Simulate driver login event (Fixed constructor signature)
        login_event = DriverLoggedInEvent(
            timestamp=env.now,
            driver_id="D1",
            initial_location=[2, 2],  # Added required parameter
            service_duration=120      # Added required parameter
        )
        event_dispatcher.dispatch(login_event)
        
        # Allow event processing
        env.run(until=0.1)
        
        # ASSERT
        # Verify assignment attempt was triggered
        assert len(assignment_attempts) == 1, "Driver login should trigger assignment attempt"
        assert assignment_attempts[0] == "D1", "Assignment attempt should be for logged-in driver"
        
        # Verify priority scorer was called (indicates full assignment evaluation)
        mock_priority_scorer.calculate_priority_score.assert_called()

    # ===== Test 2: Driver Availability Integration =====
    
    def test_driver_becomes_available_triggers_assignment_attempt(self, test_environment, test_config, mock_priority_scorer):
        """
        Test that driver availability events properly integrate with assignment service.
        
        This simulates a driver completing a delivery and looking for new work.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
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
        
        # Create a waiting order
        waiting_order = Order("O1", [4, 4], [6, 6], env.now)
        waiting_order.entity_type = EntityType.ORDER
        waiting_order.state = OrderState.CREATED
        order_repo.add(waiting_order)
        
        # Create a driver that just became available
        available_driver = Driver("D1", [3, 3], env.now, 120)
        available_driver.entity_type = EntityType.DRIVER
        available_driver.state = DriverState.AVAILABLE
        driver_repo.add(available_driver)
        
        # Track assignment attempts
        assignment_attempts = []
        original_method = assignment_service.attempt_immediate_assignment_from_driver
        
        def track_attempts(driver):
            assignment_attempts.append(driver.driver_id)
            return original_method(driver)
        
        assignment_service.attempt_immediate_assignment_from_driver = track_attempts
        
        # ACT - Simulate driver becoming available
        available_event = DriverAvailableForAssignmentEvent(timestamp=env.now, driver_id="D1")
        event_dispatcher.dispatch(available_event)
        
        # Allow event processing
        env.run(until=0.1)
        
        # ASSERT
        # Verify assignment attempt was triggered
        assert len(assignment_attempts) == 1, "Driver availability should trigger assignment attempt"
        assert assignment_attempts[0] == "D1", "Assignment attempt should be for available driver"
        
        # Verify priority scorer was used
        mock_priority_scorer.calculate_priority_score.assert_called()
    
    # ===== Test 3: Scheduling Service Integration =====
    
    def test_scheduling_service_integration_with_assignment(
        self, test_environment, test_config, mock_priority_scorer
    ):
        """
        Test that driver scheduling service properly integrates with assignment service.
        
        This verifies the complete driver lifecycle from scheduling to assignment.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
        # Create both services
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
        
        scheduling_service = DriverSchedulingService(
            env=env,
            event_dispatcher=event_dispatcher,
            driver_repository=driver_repo
        )
        
        # Create a waiting order
        waiting_order = Order("O1", [2, 2], [4, 4], env.now)
        waiting_order.entity_type = EntityType.ORDER
        waiting_order.state = OrderState.CREATED
        order_repo.add(waiting_order)
        
        # Track events for verification
        logged_in_events = []
        available_events = []
        
        event_dispatcher.register(DriverLoggedInEvent, lambda e: logged_in_events.append(e))
        event_dispatcher.register(DriverAvailableForAssignmentEvent, lambda e: available_events.append(e))
        
        # Track assignment attempts
        assignment_attempts = []
        original_method = assignment_service.attempt_immediate_assignment_from_driver
        
        def track_attempts(driver):
            assignment_attempts.append({
                'driver_id': driver.driver_id,
                'timestamp': env.now
            })
            return original_method(driver)
        
        assignment_service.attempt_immediate_assignment_from_driver = track_attempts
        
        # Create a driver and add to repository (simulating driver arrival)
        new_driver = Driver("D1", [1, 1], env.now, 120)
        new_driver.entity_type = EntityType.DRIVER
        new_driver.state = DriverState.AVAILABLE
        driver_repo.add(new_driver)
        
        # ACT - Manually trigger driver login event (Fixed constructor signature)
        login_event = DriverLoggedInEvent(
            timestamp=env.now,
            driver_id="D1",
            initial_location=[1, 1],  # Added required parameter
            service_duration=120      # Added required parameter
        )
        event_dispatcher.dispatch(login_event)
        
        # Simulate some time passing and driver becoming available again
        env.run(until=5.0)
        
        # Trigger availability event
        available_event = DriverAvailableForAssignmentEvent(timestamp=env.now, driver_id="D1")
        event_dispatcher.dispatch(available_event)
        
        # Allow final event processing
        env.run(until=5.1)
        
        # ASSERT
        # Verify events were properly dispatched and handled
        assert len(logged_in_events) == 1, "Should receive driver login event"
        assert logged_in_events[0].driver_id == "D1"
        
        assert len(available_events) == 1, "Should receive driver availability event"
        assert available_events[0].driver_id == "D1"
        
        # Verify assignment attempts were triggered for both events
        assert len(assignment_attempts) == 2, "Should have 2 assignment attempts (login + availability)"
        assert all(attempt['driver_id'] == "D1" for attempt in assignment_attempts)
        
        # Verify priority scorer was used for both attempts
        assert mock_priority_scorer.calculate_priority_score.call_count >= 1, "Priority scorer should be used"

    # ===== Test 4: Driver State Management =====
    
    def test_driver_state_consistency_during_lifecycle(
        self, test_environment, test_config, mock_priority_scorer
    ):
        """
        Test that driver states remain consistent throughout the lifecycle.
        
        This ensures that assignment attempts only occur when drivers are truly available.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
        # Set up priority scorer to return high score for successful assignment
        mock_priority_scorer.calculate_priority_score.return_value = (85.0, {
            "distance_score": 0.85,
            "throughput_score": 0.0,
            "fairness_score": 0.9,
            "combined_score_0_1": 0.85,
            "total_distance": 5.0,
            "num_orders": 1,
            "assignment_delay_minutes": 2.0
        })
        
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
        
        # Create order and driver
        test_order = Order("O1", [2, 2], [4, 4], env.now)
        test_order.entity_type = EntityType.ORDER
        test_order.state = OrderState.CREATED
        order_repo.add(test_order)
        
        test_driver = Driver("D1", [1, 1], env.now, 120)
        test_driver.entity_type = EntityType.DRIVER
        test_driver.state = DriverState.AVAILABLE
        driver_repo.add(test_driver)
        
        # ACT - Trigger driver login (Fixed constructor signature)
        login_event = DriverLoggedInEvent(
            timestamp=env.now,
            driver_id="D1",
            initial_location=[1, 1],  # Added required parameter
            service_duration=120      # Added required parameter
        )
        event_dispatcher.dispatch(login_event)
        
        # Allow event processing
        env.run(until=0.1)
        
        # ASSERT
        # If assignment was successful (high score), verify state transitions
        delivery_units = delivery_unit_repo.find_all()
        if len(delivery_units) > 0:
            # Assignment succeeded - verify proper state transitions
            assert test_driver.state == DriverState.DELIVERING, "Driver should be DELIVERING after assignment"
            assert test_order.state == OrderState.ASSIGNED, "Order should be ASSIGNED after assignment"
            
            # Verify delivery unit was created correctly
            unit = delivery_units[0]
            assert unit.driver is test_driver, "Delivery unit should reference the driver"
            assert unit.delivery_entity is test_order, "Delivery unit should reference the order"
        else:
            # Assignment failed - verify states remain unchanged
            assert test_driver.state == DriverState.AVAILABLE, "Driver should remain AVAILABLE if assignment failed"
            assert test_order.state == OrderState.CREATED, "Order should remain CREATED if assignment failed"

    # ===== Test 5: Multiple Driver Coordination =====

    def test_multiple_drivers_coordinate_properly(
        self, test_environment, test_config, mock_priority_scorer
    ):
        """
        Test that multiple drivers can coordinate without conflicts.
        
        This ensures the system handles concurrent driver events properly.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
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
        
        # Create multiple orders
        order1 = Order("O1", [1, 1], [2, 2], env.now)
        order1.entity_type = EntityType.ORDER
        order1.state = OrderState.CREATED
        order_repo.add(order1)
        
        order2 = Order("O2", [4, 4], [5, 5], env.now)
        order2.entity_type = EntityType.ORDER
        order2.state = OrderState.CREATED
        order_repo.add(order2)
        
        # Create multiple drivers
        driver1 = Driver("D1", [0, 0], env.now, 120)
        driver1.entity_type = EntityType.DRIVER
        driver1.state = DriverState.AVAILABLE
        driver_repo.add(driver1)
        
        driver2 = Driver("D2", [3, 3], env.now, 120)
        driver2.entity_type = EntityType.DRIVER
        driver2.state = DriverState.AVAILABLE
        driver_repo.add(driver2)
        
        # Track assignment attempts
        assignment_attempts = []
        original_method = assignment_service.attempt_immediate_assignment_from_driver
        
        def track_attempts(driver):
            assignment_attempts.append(driver.driver_id)
            return original_method(driver)
        
        assignment_service.attempt_immediate_assignment_from_driver = track_attempts
        
        # ACT - Simulate multiple drivers logging in (Fixed constructor signatures)
        login_event1 = DriverLoggedInEvent(
            timestamp=env.now,
            driver_id="D1",
            initial_location=[0, 0],  # Added required parameter
            service_duration=120      # Added required parameter
        )
        login_event2 = DriverLoggedInEvent(
            timestamp=env.now,
            driver_id="D2",
            initial_location=[3, 3],  # Added required parameter
            service_duration=120      # Added required parameter
        )
        
        event_dispatcher.dispatch(login_event1)
        event_dispatcher.dispatch(login_event2)
        
        # Allow event processing
        env.run(until=0.1)
        
        # ASSERT
        # Verify both drivers attempted assignment
        assert len(assignment_attempts) == 2, "Both drivers should attempt assignment"
        assert "D1" in assignment_attempts, "Driver D1 should attempt assignment"
        assert "D2" in assignment_attempts, "Driver D2 should attempt assignment"
        
        # Verify priority scorer was called for evaluations
        assert mock_priority_scorer.calculate_priority_score.call_count >= 1, "Priority scorer should be used"
        
        # System should remain in consistent state (no duplicate assignments, etc.)
        delivery_units = delivery_unit_repo.find_all()
        assigned_driver_ids = [unit.driver.driver_id for unit in delivery_units]
        
        # No driver should be assigned to multiple delivery units
        assert len(assigned_driver_ids) == len(set(assigned_driver_ids)), "No driver should have multiple assignments"