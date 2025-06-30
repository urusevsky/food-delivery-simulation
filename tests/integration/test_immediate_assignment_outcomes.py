# tests/integration/test_immediate_assignment_outcomes.py
"""
Integration tests for Area 3: Immediate Assignment Decision Outcomes (Priority Scoring)

This test file focuses on the possible outcomes of immediate assignment attempts
once they've been triggered. We test the decision logic that determines whether
an assignment succeeds or fails, and verify the proper state changes and events
that result from these decisions.

Key principle: These tests assume assignment attempts have been triggered
(Area 2's responsibility) and focus on what happens next.

Updated for priority scoring system:
- AssignmentService now requires priority_scorer parameter
- Tests verify score-based decisions instead of cost-based
- Threshold comparisons: score >= threshold (instead of cost <= threshold)
- Assignment data stored as assignment_scores (instead of assignment_costs)
"""

import pytest
import simpy
from unittest.mock import Mock

# Import core components
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.events.delivery_unit_events import DeliveryUnitAssignedEvent
from delivery_sim.repositories.order_repository import OrderRepository
from delivery_sim.repositories.driver_repository import DriverRepository
from delivery_sim.repositories.pair_repository import PairRepository
from delivery_sim.repositories.delivery_unit_repository import DeliveryUnitRepository
from delivery_sim.services.assignment_service import AssignmentService
from delivery_sim.entities.order import Order
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.pair import Pair
from delivery_sim.entities.delivery_unit import DeliveryUnit
from delivery_sim.entities.states import OrderState, DriverState, PairState
from delivery_sim.utils.entity_type_utils import EntityType
from delivery_sim.utils.location_utils import calculate_distance


class TestImmediateAssignmentOutcomes:
    """
    Test suite verifying all possible outcomes when immediate assignment is attempted.
    
    Updated for priority scoring system - tests score-based decision outcomes.
    """
    
    @pytest.fixture
    def test_config_standard_threshold(self):
        """Configuration with standard assignment threshold for testing normal operations."""
        class TestConfig:
            def __init__(self):
                # Priority scoring threshold (0-100 scale)
                self.immediate_assignment_threshold = 75.0
                self.periodic_interval = 10.0
                self.pairing_enabled = True
                self.driver_speed = 0.5
        
        return TestConfig()
    
    @pytest.fixture
    def test_config_low_threshold(self):
        """Configuration with very low threshold to test assignment failures."""
        class TestConfig:
            def __init__(self):
                # Very high threshold - most assignments should fail
                self.immediate_assignment_threshold = 95.0
                self.periodic_interval = 10.0
                self.pairing_enabled = True
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
        """Create a mock priority scorer for assignment service."""
        scorer = Mock()
        # Default return: decent score that might pass thresholds
        scorer.calculate_priority_score.return_value = (80.0, {
            "distance_score": 0.8,
            "throughput_score": 0.5,
            "fairness_score": 0.9,
            "combined_score_0_1": 0.80,
            "total_distance": 6.5,
            "num_orders": 1,
            "assignment_delay_minutes": 5.0
        })
        return scorer
    
    # ===== Outcome 1: No Available Resources =====
    
    def test_no_drivers_available_for_entity(self, test_environment, test_config_standard_threshold, mock_priority_scorer):
        """
        Test immediate assignment when no drivers are available.
        
        This simulates the common scenario where orders arrive but no drivers
        are currently free to serve them. The order should remain unassigned
        and available for future assignment attempts.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_standard_threshold
        
        # Create the assignment service with priority scorer
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
        
        # Create a test order that needs assignment
        test_order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[7, 7],
            arrival_time=env.now
        )
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        # Important: No drivers in the system!
        
        # ACT - Attempt assignment
        result = assignment_service.attempt_immediate_assignment_from_delivery_entity(test_order)
        
        # ASSERT
        # Verify the assignment failed due to no drivers
        assert result is False, "Assignment should fail when no drivers are available"
        
        # Verify no delivery unit was created
        assert len(delivery_unit_repo.find_all()) == 0, "No delivery unit should be created"
        
        # Verify order remains in original state
        assert test_order.state == OrderState.CREATED, "Order should remain in CREATED state"
        
        # Priority scorer should not have been called since no drivers to evaluate
        mock_priority_scorer.calculate_priority_score.assert_not_called()
    
    # ===== Outcome 2: Score Below Threshold =====
    
    def test_assignment_fails_when_score_below_threshold(self, test_environment, test_config_low_threshold, mock_priority_scorer):
        """
        Test immediate assignment failure when priority score is below threshold.
        
        This tests the core decision logic of the immediate assignment algorithm.
        Even with available resources, assignment should be deferred if the score
        is too low, allowing for better matches through periodic optimization.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_low_threshold  # High threshold (95.0) to ensure failure
        
        # Set up priority scorer to return low score
        low_score = 70.0  # Below the 95.0 threshold
        mock_priority_scorer.calculate_priority_score.return_value = (low_score, {
            "distance_score": 0.7,
            "throughput_score": 0.0,
            "fairness_score": 0.7,
            "combined_score_0_1": 0.70,
            "total_distance": 12.0,
            "num_orders": 1,
            "assignment_delay_minutes": 8.0
        })
        
        # Create the assignment service
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
        
        # Create an order
        test_order = Order(
            order_id="O1",
            restaurant_location=[1, 1],
            customer_location=[2, 2],
            arrival_time=env.now - 5  # 5 minutes old
        )
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        # Create a driver
        test_driver = Driver(
            driver_id="D1",
            initial_location=[9, 9],  # Far location to justify low score
            login_time=env.now,
            service_duration=120
        )
        test_driver.entity_type = EntityType.DRIVER
        test_driver.state = DriverState.AVAILABLE
        driver_repo.add(test_driver)
        
        # ACT - Attempt assignment
        result = assignment_service.attempt_immediate_assignment_from_delivery_entity(test_order)
        
        # ASSERT
        # Verify the assignment failed
        assert result is False, "Assignment should fail when score is below threshold"
        
        # Verify priority score was calculated and below threshold
        mock_priority_scorer.calculate_priority_score.assert_called_once()
        calculated_score = mock_priority_scorer.calculate_priority_score.return_value[0]
        assert calculated_score < config.immediate_assignment_threshold, \
            f"Calculated score {calculated_score} should be below threshold {config.immediate_assignment_threshold}"
        
        # Verify no delivery unit was created
        assert len(delivery_unit_repo.find_all()) == 0, "No delivery unit should be created"
        
        # Verify entities remain in their original states
        assert test_order.state == OrderState.CREATED, "Order should remain in CREATED state"
        assert test_driver.state == DriverState.AVAILABLE, "Driver should remain AVAILABLE"
    
    # ===== Outcome 3: Successful Assignment =====
    
    def test_successful_assignment_creates_delivery_unit(self, test_environment, test_config_standard_threshold, mock_priority_scorer):
        """
        Test successful assignment when priority score meets threshold.
        
        This verifies the complete assignment flow when conditions are favorable,
        including proper state transitions and event generation.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_standard_threshold
        
        # Set up priority scorer to return high score
        high_score = 85.0  # Above the 75.0 threshold
        score_components = {
            "distance_score": 0.9,
            "throughput_score": 0.0,
            "fairness_score": 0.8,
            "combined_score_0_1": 0.85,
            "total_distance": 5.0,
            "num_orders": 1,
            "assignment_delay_minutes": 3.0
        }
        mock_priority_scorer.calculate_priority_score.return_value = (high_score, score_components)
        
        # Track DeliveryUnitAssignedEvent
        assigned_events = []
        event_dispatcher.register(DeliveryUnitAssignedEvent, lambda e: assigned_events.append(e))
        
        # Create the assignment service
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
        
        # Create a test order
        test_order = Order(
            order_id="O1",
            restaurant_location=[2, 2],
            customer_location=[4, 4],
            arrival_time=env.now - 3
        )
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        # Create a close driver for good score
        close_driver = Driver(
            driver_id="D1",
            initial_location=[1, 1],  # Close to restaurant
            login_time=env.now,
            service_duration=120
        )
        close_driver.entity_type = EntityType.DRIVER
        close_driver.state = DriverState.AVAILABLE
        driver_repo.add(close_driver)
        
        # ACT - Attempt assignment
        result = assignment_service.attempt_immediate_assignment_from_delivery_entity(test_order)
        
        # Run briefly to allow event processing
        env.run(until=0.1)
        
        # ASSERT
        # Verify the assignment succeeded
        assert result is True, "Assignment should succeed when score meets threshold"
        
        # Verify score was above threshold
        mock_priority_scorer.calculate_priority_score.assert_called_once()
        calculated_score = mock_priority_scorer.calculate_priority_score.return_value[0]
        assert calculated_score >= config.immediate_assignment_threshold, \
            f"Calculated score {calculated_score} should meet threshold {config.immediate_assignment_threshold}"
        
        # Verify delivery unit was created
        delivery_units = delivery_unit_repo.find_all()
        assert len(delivery_units) == 1, "Exactly one delivery unit should be created"
        
        created_unit = delivery_units[0]
        assert created_unit.delivery_entity is test_order, "Delivery unit should reference the order"
        assert created_unit.driver is close_driver, "Delivery unit should reference the driver"
        assert created_unit.assignment_path == "immediate", "Should be marked as immediate assignment"
        
        # Verify score components were stored (updated for priority scoring)
        assert created_unit.assignment_scores is not None, "Score components should be stored"
        assert created_unit.assignment_scores["priority_score_0_100"] == high_score
        assert created_unit.assignment_scores["distance_score"] == score_components["distance_score"]
        assert created_unit.assignment_scores["total_distance"] == score_components["total_distance"]
        
        # Verify proper state transitions
        assert test_order.state == OrderState.ASSIGNED, "Order should be ASSIGNED"
        assert close_driver.state == DriverState.DELIVERING, "Driver should be DELIVERING"
        
        # Verify event was dispatched
        assert len(assigned_events) == 1, "Should dispatch one assignment event"
        event = assigned_events[0]
        assert event.entity_id == test_order.order_id
        assert event.driver_id == close_driver.driver_id
    
    # ===== Score Calculation Verification =====
    
    def test_priority_score_calculation_integration(self, test_environment, test_config_standard_threshold, mock_priority_scorer):
        """
        Test that priority score calculation integrates correctly with assignment decisions.
        
        This test verifies that the assignment service correctly uses the priority scorer
        and makes decisions based on the returned scores and components.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_standard_threshold
        
        # Create detailed score calculation tracking
        score_calculations = []
        
        def track_score_calc(driver, entity):
            original_result = (78.5, {
                "distance_score": 0.8,
                "throughput_score": 0.0,
                "fairness_score": 0.95,
                "combined_score_0_1": 0.785,
                "total_distance": 6.2,
                "num_orders": 1,
                "assignment_delay_minutes": 2.0
            })
            score_calculations.append({
                'driver_id': driver.driver_id,
                'entity_id': entity.order_id if hasattr(entity, 'order_id') else entity.pair_id,
                'priority_score': original_result[0],
                'components': original_result[1]
            })
            return original_result
        
        mock_priority_scorer.calculate_priority_score.side_effect = track_score_calc
        
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
        
        # Create test entities
        test_order = Order(
            order_id="O1",
            restaurant_location=[2, 2],
            customer_location=[5, 5],
            arrival_time=env.now - 2
        )
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        test_driver = Driver(
            driver_id="D1",
            initial_location=[0, 0],
            login_time=env.now,
            service_duration=120
        )
        test_driver.entity_type = EntityType.DRIVER
        test_driver.state = DriverState.AVAILABLE
        driver_repo.add(test_driver)
        
        # ACT - Attempt assignment to trigger score calculation
        result = assignment_service.attempt_immediate_assignment_from_delivery_entity(test_order)
        
        # ASSERT
        # Verify score was calculated
        assert len(score_calculations) == 1, "Score should be calculated once"
        
        calc = score_calculations[0]
        components = calc['components']
        
        # Verify all components are present and reasonable
        assert 0.0 <= components['distance_score'] <= 1.0, "Distance score should be in [0,1] range"
        assert 0.0 <= components['throughput_score'] <= 1.0, "Throughput score should be in [0,1] range"
        assert 0.0 <= components['fairness_score'] <= 1.0, "Fairness score should be in [0,1] range"
        assert 0.0 <= components['combined_score_0_1'] <= 1.0, "Combined score should be in [0,1] range"
        assert components['total_distance'] > 0, "Total distance should be positive"
        assert components['num_orders'] == 1, "Should be 1 order for single order"
        assert components['assignment_delay_minutes'] >= 0, "Wait time should be non-negative"
        
        # Verify assignment succeeded (score 78.5 >= threshold 75.0)
        assert result is True, "Assignment should succeed with this score"
        
        # If assignment succeeded, verify scores were stored in delivery unit
        delivery_units = delivery_unit_repo.find_all()
        if len(delivery_units) > 0:
            unit = delivery_units[0]
            assert unit.assignment_scores['priority_score_0_100'] == calc['priority_score']
            assert unit.assignment_scores['distance_score'] == components['distance_score']
            assert unit.assignment_scores['total_distance'] == components['total_distance']