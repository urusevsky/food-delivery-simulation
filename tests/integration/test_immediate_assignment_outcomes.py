# tests/integration/test_immediate_assignment_outcomes.py
"""
Integration tests for Area 3: Immediate Assignment Decision Outcomes

This test file focuses on the possible outcomes of immediate assignment attempts
once they've been triggered. We test the decision logic that determines whether
an assignment succeeds or fails, and verify the proper state changes and events
that result from these decisions.

Key principle: These tests assume assignment attempts have been triggered
(Area 2's responsibility) and focus on what happens next.
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
    
    We test four main outcome scenarios:
    1. No available resources (no drivers when entity needs one, no entities when driver needs one)
    2. Cost exceeds threshold (assignment is too expensive)
    3. Successful assignment (cost within threshold)
    4. Proper state updates and event generation
    
    These tests cover both perspectives:
    - Delivery entity (order/pair) looking for a driver
    - Driver looking for a delivery entity
    """
    
    @pytest.fixture
    def test_config_standard_threshold(self):
        """Configuration with standard assignment threshold for testing normal operations."""
        class TestConfig:
            def __init__(self):
                # Assignment parameters
                self.immediate_assignment_threshold = 5.0  # Reasonable threshold
                self.periodic_interval = 10.0
                self.throughput_factor = 1.0  # 1 km discount per order
                self.age_factor = 0.1  # 0.1 km discount per minute waiting
                self.driver_speed = 0.5  # km per minute
                
                # Pairing configuration
                self.pairing_enabled = False  # Simplified for outcome testing
        
        return TestConfig()
    
    @pytest.fixture
    def test_config_low_threshold(self):
        """Configuration with very low threshold to test assignment failures."""
        class TestConfig:
            def __init__(self):
                # Assignment parameters with restrictive threshold
                self.immediate_assignment_threshold = 1.0  # Very low - most assignments will fail
                self.periodic_interval = 10.0
                self.throughput_factor = 1.0
                self.age_factor = 0.1
                self.driver_speed = 0.5
                
                # Pairing configuration
                self.pairing_enabled = False
        
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
    
    # ===== Outcome 1: No Available Resources =====
    
    def test_assignment_fails_when_no_drivers_available(self, test_environment, test_config_standard_threshold):
        """
        Test that assignment fails gracefully when no drivers are available.
        
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
        # Verify our test setup
        assert len(driver_repo.find_available_drivers()) == 0, "Test setup: No drivers should be available"
        
        # Track assignment outcome
        original_method = assignment_service.attempt_immediate_assignment_from_delivery_entity
        assignment_results = []
        
        def track_result(entity):
            result = original_method(entity)
            assignment_results.append(result)
            return result
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = track_result
        
        # Track DeliveryUnitAssignedEvent (shouldn't be dispatched)
        assigned_events = []
        event_dispatcher.register(DeliveryUnitAssignedEvent, lambda e: assigned_events.append(e))
        
        # ACT - Attempt assignment for the order
        result = assignment_service.attempt_immediate_assignment_from_delivery_entity(test_order)
        
        # ASSERT
        # Verify the assignment failed
        assert result is False, "Assignment should fail when no drivers are available"
        assert len(assignment_results) == 1 and assignment_results[0] is False
        
        # Verify no delivery unit was created
        assert len(delivery_unit_repo.find_all()) == 0, "No delivery unit should be created"
        
        # Verify no assignment event was dispatched
        assert len(assigned_events) == 0, "No DeliveryUnitAssignedEvent should be dispatched"
        
        # Verify order state remained unchanged
        assert test_order.state == OrderState.CREATED, "Order should remain in CREATED state"
        assert test_order.delivery_unit is None, "Order should have no delivery unit reference"
    
    def test_assignment_fails_when_no_entities_waiting(self, test_environment, test_config_standard_threshold):
        """
        Test that assignment fails gracefully when a driver has no entities to deliver.
        
        This simulates the scenario where a driver becomes available but there are
        no orders or pairs waiting for assignment. The driver should remain available
        for future orders.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_standard_threshold
        
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
        
        # Create an available driver
        test_driver = Driver(
            driver_id="D1",
            initial_location=[5, 5],
            login_time=env.now,
            service_duration=120
        )
        test_driver.entity_type = EntityType.DRIVER
        test_driver.state = DriverState.AVAILABLE
        driver_repo.add(test_driver)
        
        # Important: No orders or pairs in the system!
        # Verify our test setup
        assert len(order_repo.find_unassigned_orders()) == 0, "Test setup: No orders should be waiting"
        assert len(pair_repo.find_unassigned_pairs()) == 0, "Test setup: No pairs should be waiting"
        
        # Track assignment outcome
        assignment_results = []
        original_method = assignment_service.attempt_immediate_assignment_from_driver
        
        def track_result(driver):
            result = original_method(driver)
            assignment_results.append(result)
            return result
        
        assignment_service.attempt_immediate_assignment_from_driver = track_result
        
        # Track DeliveryUnitAssignedEvent (shouldn't be dispatched)
        assigned_events = []
        event_dispatcher.register(DeliveryUnitAssignedEvent, lambda e: assigned_events.append(e))
        
        # ACT - Attempt assignment from driver perspective
        result = assignment_service.attempt_immediate_assignment_from_driver(test_driver)
        
        # ASSERT
        # Verify the assignment failed
        assert result is False, "Assignment should fail when no delivery entities are waiting"
        assert len(assignment_results) == 1 and assignment_results[0] is False
        
        # Verify no delivery unit was created
        assert len(delivery_unit_repo.find_all()) == 0, "No delivery unit should be created"
        
        # Verify no assignment event was dispatched
        assert len(assigned_events) == 0, "No DeliveryUnitAssignedEvent should be dispatched"
        
        # Verify driver state remained unchanged
        assert test_driver.state == DriverState.AVAILABLE, "Driver should remain AVAILABLE"
        assert test_driver.current_delivery_unit is None, "Driver should have no delivery unit"
    
    # ===== Outcome 2: Cost Exceeds Threshold =====
    
    def test_assignment_fails_when_cost_exceeds_threshold(self, test_environment, test_config_low_threshold):
        """
        Test that assignment fails when the adjusted cost exceeds the immediate threshold.
        
        This tests the core decision logic of the immediate assignment algorithm.
        Even with available resources, assignment should be deferred if the cost
        is too high, allowing for better matches through periodic optimization.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_low_threshold  # Using low threshold to ensure failure
        
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
        
        # Create an order at one location
        test_order = Order(
            order_id="O1",
            restaurant_location=[1, 1],
            customer_location=[2, 2],
            arrival_time=env.now - 5  # 5 minutes old for some age discount
        )
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        # Create a driver far from the order to ensure high cost
        far_driver = Driver(
            driver_id="D1",
            initial_location=[9, 9],  # Far from order's restaurant
            login_time=env.now,
            service_duration=120
        )
        far_driver.entity_type = EntityType.DRIVER
        far_driver.state = DriverState.AVAILABLE
        driver_repo.add(far_driver)
        
        # Calculate expected cost to verify our setup creates high cost
        driver_to_restaurant = calculate_distance([9, 9], [1, 1])  # Should be ~11.3 km
        restaurant_to_customer = calculate_distance([1, 1], [2, 2])  # Should be ~1.4 km
        base_cost = driver_to_restaurant + restaurant_to_customer  # ~12.7 km
        
        # With our config: adjusted_cost = base_cost - 1.0 (throughput) - 0.5 (age)
        # Expected adjusted cost ~11.2 km, which exceeds threshold of 1.0
        
        # Track cost calculations for verification
        cost_calculations = []
        original_calc_method = assignment_service.calculate_adjusted_cost
        
        def track_cost_calc(driver, entity):
            adjusted_cost, components = original_calc_method(driver, entity)
            cost_calculations.append({
                'adjusted_cost': adjusted_cost,
                'components': components
            })
            return adjusted_cost, components
        
        assignment_service.calculate_adjusted_cost = track_cost_calc
        
        # Track assignment outcome
        assignment_results = []
        original_attempt_method = assignment_service.attempt_immediate_assignment_from_delivery_entity
        
        def track_result(entity):
            result = original_attempt_method(entity)
            assignment_results.append(result)
            return result
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = track_result
        
        # ACT - Attempt assignment
        result = assignment_service.attempt_immediate_assignment_from_delivery_entity(test_order)
        
        # ASSERT
        # Verify the assignment failed
        assert result is False, "Assignment should fail when cost exceeds threshold"
        
        # Verify cost was calculated and exceeded threshold
        assert len(cost_calculations) > 0, "Cost should have been calculated"
        calculated_cost = cost_calculations[0]['adjusted_cost']
        assert calculated_cost > config.immediate_assignment_threshold, \
            f"Calculated cost {calculated_cost} should exceed threshold {config.immediate_assignment_threshold}"
        
        # Verify no delivery unit was created
        assert len(delivery_unit_repo.find_all()) == 0, "No delivery unit should be created"
        
        # Verify entities remain in their original states
        assert test_order.state == OrderState.CREATED, "Order should remain in CREATED state"
        assert far_driver.state == DriverState.AVAILABLE, "Driver should remain AVAILABLE"
    
    # ===== Outcome 3: Successful Assignment =====
    
    def test_successful_assignment_creates_delivery_unit(self, test_environment, test_config_standard_threshold):
        """
        Test successful assignment when cost is within threshold.
        
        This is the "happy path" where all conditions are met for immediate assignment.
        We verify that all state updates occur correctly and the proper events are
        dispatched to coordinate with other services.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_standard_threshold
        
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
        
        # Create an order with favorable positioning
        test_order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[3.5, 3.5],  # Very close to restaurant (0.7 km)
            arrival_time=env.now - 10  # 10 minutes old for age discount
        )
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        # Create a driver close to the order
        close_driver = Driver(
            driver_id="D1",
            initial_location=[3.2, 3.2],  # Very close to restaurant (0.28 km)
            login_time=env.now,
            service_duration=120
        )
        close_driver.entity_type = EntityType.DRIVER
        close_driver.state = DriverState.AVAILABLE
        driver_repo.add(close_driver)
        
        # Calculate expected costs for verification
        # Base cost should be ~1.0 km total (0.28 + 0.7)
        # Adjusted cost = 1.0 - 1.0 (throughput) - 1.0 (age) = -1.0 (negative is good!)
        # This should easily meet the threshold of 5.0
        
        # Track DeliveryUnitAssignedEvent
        assigned_events = []
        event_dispatcher.register(DeliveryUnitAssignedEvent, lambda e: assigned_events.append(e))
        
        # Track cost calculations
        cost_calculations = []
        original_calc_method = assignment_service.calculate_adjusted_cost
        
        def track_cost_calc(driver, entity):
            adjusted_cost, components = original_calc_method(driver, entity)
            cost_calculations.append({
                'adjusted_cost': adjusted_cost,
                'components': components
            })
            return adjusted_cost, components
        
        assignment_service.calculate_adjusted_cost = track_cost_calc
        
        # ACT - Attempt assignment
        result = assignment_service.attempt_immediate_assignment_from_delivery_entity(test_order)
        
        # Run briefly to allow event processing
        env.run(until=0.1)
        
        # ASSERT
        # Verify the assignment succeeded
        assert result is True, "Assignment should succeed when cost is within threshold"
        
        # Verify cost was within threshold
        assert len(cost_calculations) > 0, "Cost should have been calculated"
        calculated_cost = cost_calculations[0]['adjusted_cost']
        assert calculated_cost <= config.immediate_assignment_threshold, \
            f"Calculated cost {calculated_cost} should be within threshold {config.immediate_assignment_threshold}"
        
        # Verify delivery unit was created
        delivery_units = delivery_unit_repo.find_all()
        assert len(delivery_units) == 1, "Exactly one delivery unit should be created"
        
        created_unit = delivery_units[0]
        assert created_unit.delivery_entity is test_order, "Delivery unit should reference the order"
        assert created_unit.driver is close_driver, "Delivery unit should reference the driver"
        assert created_unit.assignment_path == "immediate", "Should be marked as immediate assignment"
        
        # Verify cost components were stored
        assert created_unit.assignment_costs is not None, "Cost components should be stored"
        assert "base_cost" in created_unit.assignment_costs
        assert "adjusted_cost" in created_unit.assignment_costs
        
        # Verify entity state updates
        assert test_order.state == OrderState.ASSIGNED, "Order should be in ASSIGNED state"
        assert test_order.delivery_unit is created_unit, "Order should reference delivery unit"
        
        assert close_driver.state == DriverState.DELIVERING, "Driver should be in DELIVERING state"
        assert close_driver.current_delivery_unit is created_unit, "Driver should reference delivery unit"
        
        # Verify event was dispatched
        assert len(assigned_events) == 1, "DeliveryUnitAssignedEvent should be dispatched"
        event = assigned_events[0]
        assert event.delivery_unit_id == created_unit.unit_id
        assert event.entity_type == EntityType.ORDER
        assert event.entity_id == test_order.order_id
        assert event.driver_id == close_driver.driver_id
    
    def test_successful_pair_assignment_updates_all_orders(self, test_environment, test_config_standard_threshold):
        """
        Test that successful assignment of a pair properly updates both constituent orders.
        
        When a pair is assigned, both orders within the pair need to transition to
        ASSIGNED state and reference the same delivery unit. This test verifies that
        the assignment service handles these multiple updates correctly.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_standard_threshold
        
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
        
        # Create two orders that form a pair
        order1 = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[4, 4],
            arrival_time=env.now - 8
        )
        order1.entity_type = EntityType.ORDER
        order1.state = OrderState.PAIRED  # Already paired
        order_repo.add(order1)
        
        order2 = Order(
            order_id="O2",
            restaurant_location=[3, 3],  # Same restaurant
            customer_location=[4.5, 4.5],
            arrival_time=env.now - 5
        )
        order2.entity_type = EntityType.ORDER
        order2.state = OrderState.PAIRED  # Already paired
        order_repo.add(order2)
        
        # Create the pair
        test_pair = Pair(order1, order2, env.now - 5)
        test_pair.entity_type = EntityType.PAIR
        test_pair.state = PairState.CREATED
        test_pair.optimal_sequence = [[3, 3], [4, 4], [4.5, 4.5]]
        test_pair.optimal_cost = 2.12  # Reasonable travel distance
        pair_repo.add(test_pair)
        
        # Set bidirectional references
        order1.pair = test_pair
        order2.pair = test_pair
        
        # Create a driver close enough for successful assignment
        close_driver = Driver(
            driver_id="D1",
            initial_location=[3.1, 3.1],  # Close to restaurant
            login_time=env.now,
            service_duration=120
        )
        close_driver.entity_type = EntityType.DRIVER
        close_driver.state = DriverState.AVAILABLE
        driver_repo.add(close_driver)
        
        # Track DeliveryUnitAssignedEvent
        assigned_events = []
        event_dispatcher.register(DeliveryUnitAssignedEvent, lambda e: assigned_events.append(e))
        
        # ACT - Attempt assignment of the pair
        result = assignment_service.attempt_immediate_assignment_from_delivery_entity(test_pair)
        
        # Run briefly to allow event processing
        env.run(until=0.1)
        
        # ASSERT
        # Verify the assignment succeeded
        assert result is True, "Pair assignment should succeed"
        
        # Verify delivery unit was created
        delivery_units = delivery_unit_repo.find_all()
        assert len(delivery_units) == 1, "Exactly one delivery unit should be created"
        
        created_unit = delivery_units[0]
        assert created_unit.delivery_entity is test_pair, "Delivery unit should reference the pair"
        
        # Verify pair state update
        assert test_pair.state == PairState.ASSIGNED, "Pair should be in ASSIGNED state"
        assert test_pair.delivery_unit is created_unit, "Pair should reference delivery unit"
        
        # Verify BOTH orders were updated
        assert order1.state == OrderState.ASSIGNED, "Order 1 should be in ASSIGNED state"
        assert order1.delivery_unit is created_unit, "Order 1 should reference delivery unit"
        
        assert order2.state == OrderState.ASSIGNED, "Order 2 should be in ASSIGNED state"
        assert order2.delivery_unit is created_unit, "Order 2 should reference delivery unit"
        
        # Verify driver state update
        assert close_driver.state == DriverState.DELIVERING, "Driver should be in DELIVERING state"
        
        # Verify event was dispatched for the pair
        assert len(assigned_events) == 1, "DeliveryUnitAssignedEvent should be dispatched"
        event = assigned_events[0]
        assert event.entity_type == EntityType.PAIR
        assert event.entity_id == test_pair.pair_id
    
    # ===== Cost Calculation Verification =====
    
    def test_cost_calculation_components_are_correct(self, test_environment, test_config_standard_threshold):
        """
        Test that cost calculations properly include all components.
        
        The adjusted cost formula is:
        adjusted_cost = base_cost - (throughput_factor * num_orders) - (age_factor * age_minutes)
        
        This test verifies that all components are calculated correctly and that
        the assignment decision is based on the adjusted cost.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_standard_threshold
        
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
        
        # Create an order with known age
        age_minutes = 15
        test_order = Order(
            order_id="O1",
            restaurant_location=[2, 2],
            customer_location=[5, 5],  # 4.24 km from restaurant
            arrival_time=env.now - age_minutes
        )
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        # Create a driver at known distance
        test_driver = Driver(
            driver_id="D1",
            initial_location=[0, 0],  # 2.83 km from restaurant
            login_time=env.now,
            service_duration=120
        )
        test_driver.entity_type = EntityType.DRIVER
        test_driver.state = DriverState.AVAILABLE
        driver_repo.add(test_driver)
        
        # Calculate expected values
        expected_base_cost = calculate_distance([0, 0], [2, 2]) + calculate_distance([2, 2], [5, 5])
        # Should be 2.83 + 4.24 = 7.07 km
        
        expected_throughput_discount = config.throughput_factor * 1  # 1 order
        # Should be 1.0 * 1 = 1.0
        
        expected_age_discount = config.age_factor * age_minutes
        # Should be 0.1 * 15 = 1.5
        
        expected_adjusted_cost = expected_base_cost - expected_throughput_discount - expected_age_discount
        # Should be 7.07 - 1.0 - 1.5 = 4.57
        
        # Track cost calculations in detail
        cost_calculations = []
        original_calc_method = assignment_service.calculate_adjusted_cost
        
        def detailed_cost_tracking(driver, entity):
            adjusted_cost, components = original_calc_method(driver, entity)
            cost_calculations.append({
                'driver_id': driver.driver_id,
                'entity_id': entity.order_id if hasattr(entity, 'order_id') else entity.pair_id,
                'adjusted_cost': adjusted_cost,
                'components': components
            })
            return adjusted_cost, components
        
        assignment_service.calculate_adjusted_cost = detailed_cost_tracking
        
        # ACT - Attempt assignment to trigger cost calculation
        result = assignment_service.attempt_immediate_assignment_from_delivery_entity(test_order)
        
        # ASSERT
        # Verify cost was calculated
        assert len(cost_calculations) == 1, "Cost should be calculated once"
        
        calc = cost_calculations[0]
        components = calc['components']
        
        # Verify all components are present and correct
        assert abs(components['base_cost'] - expected_base_cost) < 0.01, \
            f"Base cost should be {expected_base_cost:.2f}, got {components['base_cost']:.2f}"
        
        assert components['num_orders'] == 1, "Should be 1 order"
        
        assert abs(components['throughput_component'] - expected_throughput_discount) < 0.01, \
            f"Throughput discount should be {expected_throughput_discount:.2f}"
        
        assert abs(components['age_minutes'] - age_minutes) < 0.01, \
            f"Age should be {age_minutes} minutes"
        
        assert abs(components['age_discount'] - expected_age_discount) < 0.01, \
            f"Age discount should be {expected_age_discount:.2f}"
        
        assert abs(calc['adjusted_cost'] - expected_adjusted_cost) < 0.01, \
            f"Adjusted cost should be {expected_adjusted_cost:.2f}, got {calc['adjusted_cost']:.2f}"
        
        # Verify assignment succeeded (adjusted cost 4.57 < threshold 5.0)
        assert result is True, "Assignment should succeed with this cost"
        
        # If assignment succeeded, verify costs were stored in delivery unit
        delivery_units = delivery_unit_repo.find_all()
        if len(delivery_units) > 0:
            unit = delivery_units[0]
            assert unit.assignment_costs['base_cost'] == components['base_cost']
            assert unit.assignment_costs['adjusted_cost'] == calc['adjusted_cost']
    
    # ===== Edge Cases and Error Handling =====
    
    def test_assignment_validates_driver_availability(self, test_environment, test_config_standard_threshold):
        """
        Test that assignment properly validates driver state before creating assignments.
        
        This is a critical safety check - even if cost calculations suggest a match,
        the assignment should fail if the driver is not actually available.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_standard_threshold
        
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
        
        # Create a test order
        test_order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[4, 4],
            arrival_time=env.now
        )
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        # Create a driver who is NOT available (already delivering)
        busy_driver = Driver(
            driver_id="D1",
            initial_location=[3, 3],  # At the restaurant
            login_time=env.now,
            service_duration=120
        )
        busy_driver.entity_type = EntityType.DRIVER
        busy_driver.state = DriverState.DELIVERING  # Not available!
        driver_repo.add(busy_driver)
        
        # The assignment service should filter out busy drivers
        # but let's test the edge case where state might have changed
        
        # Track assignment attempts
        assignment_results = []
        original_method = assignment_service.attempt_immediate_assignment_from_delivery_entity
        
        def track_result(entity):
            result = original_method(entity)
            assignment_results.append(result)
            return result
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = track_result
        
        # ACT - Attempt assignment
        result = assignment_service.attempt_immediate_assignment_from_delivery_entity(test_order)
        
        # ASSERT
        # Assignment should fail because no AVAILABLE drivers exist
        assert result is False, "Assignment should fail when driver is not available"
        
        # Verify no delivery unit was created
        assert len(delivery_unit_repo.find_all()) == 0, "No delivery unit should be created"
        
        # Verify order remains unassigned
        assert test_order.state == OrderState.CREATED, "Order should remain in CREATED state"