# tests/integration/test_pairing_assignment_integration.py
import pytest
import simpy
from unittest.mock import patch, Mock

# Import core components
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.events.order_events import OrderCreatedEvent
from delivery_sim.events.pair_events import PairCreatedEvent, PairingFailedEvent
from delivery_sim.events.driver_events import DriverLoggedInEvent
from delivery_sim.events.delivery_unit_events import DeliveryUnitAssignedEvent
from delivery_sim.repositories.order_repository import OrderRepository
from delivery_sim.repositories.driver_repository import DriverRepository
from delivery_sim.repositories.pair_repository import PairRepository
from delivery_sim.repositories.delivery_unit_repository import DeliveryUnitRepository
from delivery_sim.services.assignment_service import AssignmentService
from delivery_sim.entities.order import Order
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.pair import Pair
from delivery_sim.entities.states import OrderState, DriverState, PairState
from delivery_sim.utils.entity_type_utils import EntityType


class TestPairingAssignmentIntegration:
    """
    Integration tests for interactions between the Pairing and Assignment services.
    
    These tests verify:
    1. OrderCreatedEvent triggers assignment when pairing is disabled
    2. PairCreatedEvent triggers assignment when pairing is enabled
    3. PairingFailedEvent triggers assignment for unpaired orders
    4. Assignment outcomes under different conditions (no drivers, cost too high, successful)
    """
    
    @pytest.fixture
    def test_config_pairing_disabled(self):
        """Create a configuration with pairing disabled."""
        class TestConfig:
            def __init__(self):
                # Assignment parameters
                self.immediate_assignment_threshold = 5.0  # High threshold for testing
                self.periodic_interval = 10.0
                self.throughput_factor = 1.0
                self.age_factor = 0.1
                self.driver_speed = 0.5  # km per minute
                
                # Pairing configuration (disabled)
                self.pairing_enabled = False
        
        return TestConfig()
    
    @pytest.fixture
    def test_config_pairing_enabled(self):
        """Create a configuration with pairing enabled."""
        class TestConfig:
            def __init__(self):
                # Assignment parameters (same as disabled config)
                self.immediate_assignment_threshold = 5.0  # High threshold for testing
                self.periodic_interval = 10.0
                self.throughput_factor = 1.0
                self.age_factor = 0.1
                self.driver_speed = 0.5  # km per minute
                
                # Pairing configuration (enabled)
                self.pairing_enabled = True
        
        return TestConfig()
    
    @pytest.fixture
    def test_config_low_threshold(self):
        """Create a configuration with a very low assignment threshold."""
        class TestConfig:
            def __init__(self):
                # Assignment parameters with low threshold
                self.immediate_assignment_threshold = 0.5  # Very low threshold to fail most assignments
                self.periodic_interval = 10.0
                self.throughput_factor = 1.0
                self.age_factor = 0.1
                self.driver_speed = 0.5  # km per minute
                
                # Pairing configuration (disabled for simplicity)
                self.pairing_enabled = False
        
        return TestConfig()
    
    @pytest.fixture
    def test_environment(self):
        """Set up the common test environment."""
        env = simpy.Environment()
        event_dispatcher = EventDispatcher()
        order_repo = OrderRepository()
        driver_repo = DriverRepository()
        pair_repo = PairRepository()
        delivery_unit_repo = DeliveryUnitRepository()
        
        # Return all components
        return {
            "env": env,
            "event_dispatcher": event_dispatcher,
            "order_repo": order_repo,
            "driver_repo": driver_repo,
            "pair_repo": pair_repo,
            "delivery_unit_repo": delivery_unit_repo
        }
    
    def test_order_created_triggers_assignment_without_pairing(self, test_environment, test_config_pairing_disabled):
        """
        Test that OrderCreatedEvent directly triggers assignment attempt when pairing is disabled.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_disabled
        
        # Create the assignment service - with pairing disabled
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=config
        )
        
        # Add a spy to track assignment attempts
        assignment_attempts = []
        original_attempt_method = assignment_service.attempt_immediate_assignment_from_delivery_entity
        
        def spy_attempt_assignment(entity):
            assignment_attempts.append(entity.order_id if hasattr(entity, 'order_id') else entity.pair_id)
            return original_attempt_method(entity)
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = spy_attempt_assignment
        
        # Create a test order
        new_order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[7, 7],
            arrival_time=env.now
        )
        order_repo.add(new_order)
        
        # ACT - Manually dispatch the event
        event_dispatcher.dispatch(OrderCreatedEvent(
            timestamp=env.now,
            order_id=new_order.order_id,
            restaurant_id="R1",
            restaurant_location=new_order.restaurant_location,
            customer_location=new_order.customer_location
        ))
        
        # Run briefly to process events
        env.run(until=0.1)
        
        # ASSERT
        assert len(assignment_attempts) == 1, "AssignmentService should attempt to assign the order"
        assert assignment_attempts[0] == new_order.order_id, "Assignment attempt should be for the correct order ID"
    
    def test_pair_created_triggers_assignment(self, test_environment, test_config_pairing_enabled):
        """
        Test that PairCreatedEvent triggers assignment attempt when pairing is enabled.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_enabled
        
        # Create the assignment service - with pairing enabled
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=config
        )
        
        # Add a spy to track assignment attempts
        assignment_attempts = []
        original_attempt_method = assignment_service.attempt_immediate_assignment_from_delivery_entity
        
        def spy_attempt_assignment(entity):
            assignment_attempts.append(entity.pair_id if hasattr(entity, 'pair_id') else entity.order_id)
            return original_attempt_method(entity)
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = spy_attempt_assignment
        
        # Create orders for a pair
        order1 = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[7, 7],
            arrival_time=env.now
        )
        
        order2 = Order(
            order_id="O2",
            restaurant_location=[3, 3],
            customer_location=[8, 8],
            arrival_time=env.now
        )
        
        # Create a pair
        pair = Pair(order1, order2, env.now)
        pair.pair_id = "P-O1_O2"  # Ensure consistent ID format
        pair.optimal_sequence = [order1.restaurant_location, order1.customer_location, order2.customer_location]
        pair.optimal_cost = 10.0  # Some reasonable cost value
        pair.entity_type = EntityType.PAIR
        pair_repo.add(pair)
        
        # ACT - Manually dispatch the event
        event_dispatcher.dispatch(PairCreatedEvent(
            timestamp=env.now,
            pair_id=pair.pair_id,
            order1_id=order1.order_id,
            order2_id=order2.order_id
        ))
        
        # Run briefly to process events
        env.run(until=0.1)
        
        # ASSERT
        assert len(assignment_attempts) == 1, "AssignmentService should attempt to assign the pair"
        assert assignment_attempts[0] == pair.pair_id, "Assignment attempt should be for the correct pair ID"
    
    def test_pairing_failed_triggers_assignment(self, test_environment, test_config_pairing_enabled):
        """
        Test that PairingFailedEvent triggers assignment attempt for an unpaired order.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_enabled
        
        # Create the assignment service - with pairing enabled
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=config
        )
        
        # Add a spy to track assignment attempts
        assignment_attempts = []
        original_attempt_method = assignment_service.attempt_immediate_assignment_from_delivery_entity
        
        def spy_attempt_assignment(entity):
            assignment_attempts.append(entity.order_id if hasattr(entity, 'order_id') else entity.pair_id)
            return original_attempt_method(entity)
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = spy_attempt_assignment
        
        # Create a test order
        new_order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[7, 7],
            arrival_time=env.now
        )
        new_order.entity_type = EntityType.ORDER
        order_repo.add(new_order)
        
        # ACT - Manually dispatch the event
        event_dispatcher.dispatch(PairingFailedEvent(
            timestamp=env.now,
            order_id=new_order.order_id
        ))
        
        # Run briefly to process events
        env.run(until=0.1)
        
        # ASSERT
        assert len(assignment_attempts) == 1, "AssignmentService should attempt to assign the unpaired order"
        assert assignment_attempts[0] == new_order.order_id, "Assignment attempt should be for the correct order ID"
    
    def test_assignment_fails_with_no_available_drivers(self, test_environment, test_config_pairing_disabled):
        """
        Test that assignment fails when there are no available drivers.
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
        
        # Create a test order (but no drivers)
        new_order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[7, 7],
            arrival_time=env.now
        )
        new_order.entity_type = EntityType.ORDER
        order_repo.add(new_order)
        
        # Track DeliveryUnitAssignedEvent
        delivery_unit_events = []
        event_dispatcher.register(DeliveryUnitAssignedEvent, lambda e: delivery_unit_events.append(e))
        
        # ACT - Manually dispatch the event to trigger assignment
        event_dispatcher.dispatch(OrderCreatedEvent(
            timestamp=env.now,
            order_id=new_order.order_id,
            restaurant_id="R1",
            restaurant_location=new_order.restaurant_location,
            customer_location=new_order.customer_location
        ))
        
        # Run briefly to process events
        env.run(until=0.1)
        
        # ASSERT
        assert len(delivery_unit_events) == 0, "No delivery unit should be created without available drivers"
        assert new_order.state == OrderState.CREATED, "Order should remain in CREATED state"
    
    def test_assignment_fails_when_cost_exceeds_threshold(self, test_environment, test_config_low_threshold):
        """
        Test that assignment fails when the adjusted cost exceeds the threshold.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_low_threshold  # Using config with low threshold
        
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
        new_order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[7, 7],
            arrival_time=env.now
        )
        new_order.entity_type = EntityType.ORDER
        order_repo.add(new_order)
        
        # Create a driver far from the order (to ensure high cost)
        driver = Driver(
            driver_id="D1",
            initial_location=[9, 9],  # Far from order location
            login_time=env.now,
            service_duration=120
        )
        driver.entity_type = EntityType.DRIVER
        driver.state = DriverState.AVAILABLE
        driver_repo.add(driver)
        
        # Track DeliveryUnitAssignedEvent
        delivery_unit_events = []
        event_dispatcher.register(DeliveryUnitAssignedEvent, lambda e: delivery_unit_events.append(e))
        
        # Spy on return value of attempt_immediate_assignment_from_delivery_entity
        original_attempt_method = assignment_service.attempt_immediate_assignment_from_delivery_entity
        assignment_results = []
        
        def spy_with_result(entity):
            result = original_attempt_method(entity)
            assignment_results.append(result)
            return result
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = spy_with_result
        
        # ACT - Manually dispatch the event to trigger assignment
        event_dispatcher.dispatch(OrderCreatedEvent(
            timestamp=env.now,
            order_id=new_order.order_id,
            restaurant_id="R1",
            restaurant_location=new_order.restaurant_location,
            customer_location=new_order.customer_location
        ))
        
        # Run briefly to process events
        env.run(until=0.1)
        
        # ASSERT
        assert len(assignment_results) == 1, "Assignment attempt should be made"
        assert assignment_results[0] is False, "Assignment attempt should return False"
        assert len(delivery_unit_events) == 0, "No delivery unit should be created when cost exceeds threshold"
        assert new_order.state == OrderState.CREATED, "Order should remain in CREATED state"
    
    def test_assignment_succeeds_with_cost_verification(self, test_environment, test_config_pairing_disabled):
        """
        Test that assignment succeeds and verify the cost calculation details.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config_pairing_disabled
        
        # Import calculate_distance for verification
        from delivery_sim.utils.location_utils import calculate_distance
        
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
        
        # Create a test order with customer very close to restaurant
        new_order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[3.5, 3.5],  # Customer very close to restaurant
            arrival_time=env.now - 10  # Give it some age for age discount
        )
        new_order.entity_type = EntityType.ORDER
        order_repo.add(new_order)
        
        # Create a driver close to the order
        driver = Driver(
            driver_id="D1",
            initial_location=[3.1, 3.1],  # Very close to restaurant
            login_time=env.now,
            service_duration=120
        )
        driver.entity_type = EntityType.DRIVER
        driver.state = DriverState.AVAILABLE
        driver_repo.add(driver)
        
        # Track DeliveryUnitAssignedEvent
        delivery_unit_events = []
        event_dispatcher.register(DeliveryUnitAssignedEvent, lambda e: delivery_unit_events.append(e))
        
        # Spy 1: Track assignment attempt results
        original_attempt_method = assignment_service.attempt_immediate_assignment_from_delivery_entity
        assignment_results = []
        
        def spy_attempt_assignment(entity):
            result = original_attempt_method(entity)
            assignment_results.append(result)
            return result
        
        assignment_service.attempt_immediate_assignment_from_delivery_entity = spy_attempt_assignment
        
        # Spy 2: Track cost calculation details
        cost_calculations = []
        original_calc_method = assignment_service.calculate_adjusted_cost
        
        def spy_cost_calculation(driver, entity):
            adjusted_cost, components = original_calc_method(driver, entity)
            cost_calculations.append({
                'adjusted_cost': adjusted_cost,
                'driver_id': driver.driver_id,
                'entity_id': entity.order_id if hasattr(entity, 'order_id') else entity.pair_id,
                'components': components
            })
            return adjusted_cost, components
        
        assignment_service.calculate_adjusted_cost = spy_cost_calculation
        
        # ACT - Manually dispatch the event to trigger assignment
        event_dispatcher.dispatch(OrderCreatedEvent(
            timestamp=env.now,
            order_id=new_order.order_id,
            restaurant_id="R1",
            restaurant_location=new_order.restaurant_location,
            customer_location=new_order.customer_location
        ))
        
        # Run briefly to process events
        env.run(until=0.1)
        
        # ASSERT - Part 1: Verify assignment outcome
        assert len(assignment_results) == 1, "Assignment attempt should be made"
        assert assignment_results[0] is True, "Assignment attempt should return True"
        assert len(delivery_unit_events) == 1, "A delivery unit should be created"
        assert new_order.state == OrderState.ASSIGNED, "Order should transition to ASSIGNED state"
        
        # Verify delivery unit event data
        event = delivery_unit_events[0]
        assert event.entity_id == new_order.order_id, "Event should reference the correct order"
        assert event.driver_id == driver.driver_id, "Event should reference the correct driver"
        
        # ASSERT - Part 2: Verify cost calculation details
        assert len(cost_calculations) > 0, "Cost calculation should be performed"
        calc = cost_calculations[0]  # Get the first calculation
        assert calc['driver_id'] == driver.driver_id, "Calculation should be for the correct driver"
        assert calc['entity_id'] == new_order.order_id, "Calculation should be for the correct order"
        
        # Verify the adjusted cost is within threshold
        assert calc['adjusted_cost'] <= config.immediate_assignment_threshold, \
            f"Adjusted cost {calc['adjusted_cost']} should be within threshold {config.immediate_assignment_threshold}"
        
        # Verify individual cost components
        components = calc['components']
        assert 'base_cost' in components, "Base cost should be calculated"
        assert 'throughput_component' in components, "Throughput component should be calculated"
        assert 'age_discount' in components, "Age discount should be calculated"
        
        # Calculate expected base cost and verify
        expected_base_cost = (
            calculate_distance(driver.location, new_order.restaurant_location) +
            calculate_distance(new_order.restaurant_location, new_order.customer_location)
        )
        assert abs(components['base_cost'] - expected_base_cost) < 0.01, \
            f"Base cost {components['base_cost']} should match expected value {expected_base_cost}"
        
        # Log the detailed cost breakdown for debugging
        print(f"\nCost calculation details:")
        print(f"  Base cost: {components['base_cost']}")
        print(f"  Throughput component: {components['throughput_component']}")
        print(f"  Age discount: {components['age_discount']}")
        print(f"  Final adjusted cost: {calc['adjusted_cost']}")
        print(f"  Threshold: {config.immediate_assignment_threshold}")