# tests/integration/test_assignment_delivery_integration.py
import pytest
import simpy
from unittest.mock import patch, Mock

# Import core components
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.events.delivery_unit_events import DeliveryUnitAssignedEvent, DeliveryUnitCompletedEvent
from delivery_sim.repositories.order_repository import OrderRepository
from delivery_sim.repositories.driver_repository import DriverRepository
from delivery_sim.repositories.pair_repository import PairRepository
from delivery_sim.repositories.delivery_unit_repository import DeliveryUnitRepository
from delivery_sim.services.delivery_service import DeliveryService
from delivery_sim.entities.order import Order
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.pair import Pair
from delivery_sim.entities.delivery_unit import DeliveryUnit
from delivery_sim.entities.states import OrderState, DriverState, DeliveryUnitState, PairState
from delivery_sim.utils.entity_type_utils import EntityType


class TestAssignmentDeliveryIntegration:
    """
    Integration tests for interactions between Assignment and Delivery services.
    
    These tests verify:
    1. DeliveryUnitAssignedEvent triggers the delivery process
    2. Order and driver states are updated correctly during delivery
    3. DeliveryUnitCompletedEvent is dispatched when delivery completes
    """
    
    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        class TestConfig:
            def __init__(self):
                # Delivery parameters
                self.driver_speed = 1.0  # Fast speed (1 km per minute) for faster test execution
                
                # Not needed for delivery but might be required by service constructor
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
    
    def test_delivery_process_for_single_order(self, test_environment, test_config):
        """
        Test that DeliveryUnitAssignedEvent triggers the delivery process for a single order.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
        # Create the delivery service
        delivery_service = DeliveryService(
            env=env,
            event_dispatcher=event_dispatcher,
            driver_repository=driver_repo,
            order_repository=order_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=config
        )
        
        # Create a test order
        order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[5, 5],  # 2.83 km from restaurant
            arrival_time=env.now
        )
        order.entity_type = EntityType.ORDER
        order.state = OrderState.ASSIGNED  # Order already assigned
        order_repo.add(order)
        
        # Create a driver
        driver = Driver(
            driver_id="D1",
            initial_location=[1, 1],  # 2.83 km from restaurant
            login_time=env.now,
            service_duration=120
        )
        driver.entity_type = EntityType.DRIVER
        driver.state = DriverState.DELIVERING  # Driver already in delivering state
        driver_repo.add(driver)
        
        # Create a delivery unit
        delivery_unit = DeliveryUnit(order, driver, env.now)
        delivery_unit.entity_type = EntityType.DELIVERY_UNIT
        delivery_unit_repo.add(delivery_unit)
        
        # Add references
        order.delivery_unit = delivery_unit
        driver.current_delivery_unit = delivery_unit
        
        # Track DeliveryUnitCompletedEvent
        completed_events = []
        event_dispatcher.register(DeliveryUnitCompletedEvent, lambda e: completed_events.append(e))
        
        # Add a spy to verify delivery process is started
        delivery_processes_started = []
        original_start_method = delivery_service.start_delivery
        
        def spy_start_delivery(driver, entity, delivery_unit):
            delivery_processes_started.append((driver.driver_id, entity.order_id if hasattr(entity, 'order_id') else entity.pair_id))
            return original_start_method(driver, entity, delivery_unit)
        
        delivery_service.start_delivery = spy_start_delivery
        
        # ACT - 1: Dispatch the event to trigger delivery process
        event_dispatcher.dispatch(DeliveryUnitAssignedEvent(
            timestamp=env.now,
            delivery_unit_id=delivery_unit.unit_id,
            entity_type=EntityType.ORDER,
            entity_id=order.order_id,
            driver_id=driver.driver_id
        ))
        
        # Run briefly to process event but not complete delivery
        env.run(until=0.1)
        
        # ASSERT - 1: Verify delivery process started
        assert len(delivery_processes_started) == 1, "Delivery process should be started"
        assert delivery_processes_started[0][0] == driver.driver_id, "Delivery process should be for the correct driver"
        assert delivery_processes_started[0][1] == order.order_id, "Delivery process should be for the correct order"
        
        # Calculate total delivery time
        # Driver → Restaurant = 2.83 km
        # Restaurant → Customer = 2.83 km
        # Total = 5.66 km
        # With speed = 1.0 km/min, this should take about 5.66 minutes
        # Adding a small buffer for processing time
        delivery_time = 6.0
        
        # ACT - 2: Run the simulation until delivery should be complete
        env.run(until=delivery_time)
        
        # ASSERT - 2: Verify state changes
        assert order.state == OrderState.DELIVERED, "Order should be delivered"
        assert driver.state == DriverState.AVAILABLE, "Driver should be available after delivery"
        assert delivery_unit.state == DeliveryUnitState.COMPLETED, "Delivery unit should be completed"
        
        # ASSERT - 3: Verify event generation
        assert len(completed_events) == 1, "DeliveryUnitCompletedEvent should be dispatched"
        assert completed_events[0].delivery_unit_id == delivery_unit.unit_id, "Event should reference the correct delivery unit"
        assert completed_events[0].driver_id == driver.driver_id, "Event should reference the correct driver"
    
    def test_delivery_process_for_pair(self, test_environment, test_config):
        """
        Test that DeliveryUnitAssignedEvent triggers the delivery process for a pair.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
        # Create the delivery service
        delivery_service = DeliveryService(
            env=env,
            event_dispatcher=event_dispatcher,
            driver_repository=driver_repo,
            order_repository=order_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=config
        )
        
        # Create two orders for the pair
        order1 = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[5, 5],  # 2.83 km from restaurant
            arrival_time=env.now
        )
        order1.entity_type = EntityType.ORDER
        order1.state = OrderState.ASSIGNED
        order_repo.add(order1)
        
        order2 = Order(
            order_id="O2",
            restaurant_location=[3, 3],  # Same restaurant as order1
            customer_location=[6, 6],  # 4.24 km from restaurant, 1.41 km from order1's customer
            arrival_time=env.now
        )
        order2.entity_type = EntityType.ORDER
        order2.state = OrderState.ASSIGNED
        order_repo.add(order2)
        
        # Create a pair
        pair = Pair(order1, order2, env.now)
        pair.pair_id = "P-O1_O2"
        pair.entity_type = EntityType.PAIR
        pair.state = PairState.ASSIGNED
        
        # Set optimal sequence - affects the delivery order
        # This sequence influences the total delivery time
        pair.optimal_sequence = [
            order1.restaurant_location,  # Restaurant
            order1.customer_location,    # First customer
            order2.customer_location     # Second customer
        ]
        pair.optimal_cost = 4.24  # Total distance of optimal sequence
        pair_repo.add(pair)
        
        # Create a driver
        driver = Driver(
            driver_id="D1",
            initial_location=[1, 1],  # 2.83 km from restaurant
            login_time=env.now,
            service_duration=120
        )
        driver.entity_type = EntityType.DRIVER
        driver.state = DriverState.DELIVERING
        driver_repo.add(driver)
        
        # Create a delivery unit
        delivery_unit = DeliveryUnit(pair, driver, env.now)
        delivery_unit.entity_type = EntityType.DELIVERY_UNIT
        delivery_unit_repo.add(delivery_unit)
        
        # Add references
        order1.pair = pair
        order2.pair = pair
        pair.delivery_unit = delivery_unit
        driver.current_delivery_unit = delivery_unit
        
        # Track DeliveryUnitCompletedEvent
        completed_events = []
        event_dispatcher.register(DeliveryUnitCompletedEvent, lambda e: completed_events.append(e))
        
        # Add a spy to verify delivery process is started
        delivery_processes_started = []
        original_start_method = delivery_service.start_delivery
        
        def spy_start_delivery(driver, entity, delivery_unit):
            delivery_processes_started.append((driver.driver_id, entity.pair_id if hasattr(entity, 'pair_id') else entity.order_id))
            return original_start_method(driver, entity, delivery_unit)
        
        delivery_service.start_delivery = spy_start_delivery
        
        # ACT - 1: Dispatch the event to trigger delivery process
        event_dispatcher.dispatch(DeliveryUnitAssignedEvent(
            timestamp=env.now,
            delivery_unit_id=delivery_unit.unit_id,
            entity_type=EntityType.PAIR,
            entity_id=pair.pair_id,
            driver_id=driver.driver_id
        ))
        
        # Run briefly to process event but not complete delivery
        env.run(until=0.1)
        
        # ASSERT - 1: Verify delivery process started
        assert len(delivery_processes_started) == 1, "Delivery process should be started"
        assert delivery_processes_started[0][0] == driver.driver_id, "Delivery process should be for the correct driver"
        assert delivery_processes_started[0][1] == pair.pair_id, "Delivery process should be for the correct pair"
        
        # Calculate total delivery time
        # Driver → Restaurant = 2.83 km
        # Restaurant → Customer1 → Customer2 = 4.24 km (from pair.optimal_cost)
        # Total = 7.07 km
        # With speed = 1.0 km/min, this should take about 7.07 minutes
        # Adding a small buffer for processing time
        delivery_time = 7.5
        
        # ACT - 2: Run the simulation until delivery should be complete
        env.run(until=delivery_time)
        
        # ASSERT - 2: Verify state changes
        assert order1.state == OrderState.DELIVERED, "Order 1 should be delivered"
        assert order2.state == OrderState.DELIVERED, "Order 2 should be delivered"
        assert pair.state == PairState.COMPLETED, "Pair should be completed" 
        assert driver.state == DriverState.AVAILABLE, "Driver should be available after delivery"
        assert delivery_unit.state == DeliveryUnitState.COMPLETED, "Delivery unit should be completed"
        
        # ASSERT - 3: Verify event generation
        assert len(completed_events) == 1, "DeliveryUnitCompletedEvent should be dispatched"
        assert completed_events[0].delivery_unit_id == delivery_unit.unit_id, "Event should reference the correct delivery unit"
        assert completed_events[0].driver_id == driver.driver_id, "Event should reference the correct driver"