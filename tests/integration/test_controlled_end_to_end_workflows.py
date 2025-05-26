# tests/integration/test_controlled_end_to_end_workflows.py
"""
Controlled End-to-End Workflow Tests

This test file verifies complete entity lifecycles using fully controlled scenarios.
Unlike integration tests that verify individual interfaces, these tests verify that
complete business processes work correctly from start to finish.

Key principles:
1. Use real simulation_runner.py and all real services
2. Bypass random arrival processes by manually creating entities
3. Control simulation timing to ensure workflows complete
4. Verify complete entity state progressions and timing relationships

These tests bridge the gap between interface testing and realistic simulation,
providing confidence that the mechanical aspects of the simulation work correctly
before moving to research-oriented parameter exploration.
"""

import pytest
import simpy

# Import core simulation components
from delivery_sim.simulation.simulation_runner import SimulationRunner
from delivery_sim.simulation.configuration import (
    StructuralConfig, OperationalConfig, ExperimentConfig, SimulationConfig
)
from delivery_sim.entities.order import Order
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.pair import Pair
from delivery_sim.entities.states import OrderState, DriverState, PairState, DeliveryUnitState
from delivery_sim.utils.entity_type_utils import EntityType
from delivery_sim.utils.location_utils import calculate_distance


class TestControlledEndToEndWorkflows:
    """
    Test suite for controlled end-to-end workflow verification.
    
    These tests use the complete simulation infrastructure but with manually
    created entities instead of random arrival processes. This allows us to
    verify that complete workflows execute correctly under controlled conditions.
    """
    
    @pytest.fixture
    def controlled_config(self):
        """
        Create configuration optimized for controlled testing.
        
        The key insight here is that we want very slow (or disabled) arrival
        processes so they don't interfere with our manually created entities,
        but fast operational processes so our tests complete quickly.
        """
        # Structural configuration - minimal infrastructure
        structural_config = StructuralConfig(
            delivery_area_size=10.0,  # Small area for predictable distances
            num_restaurants=3,        # Few restaurants for simplicity
            driver_speed=2.0          # Fast speed for quick test execution
        )
        
        # Operational configuration - optimized for controlled testing
        operational_config = OperationalConfig(
            # Arrival processes - set very slow so they don't interfere
            mean_order_inter_arrival_time=1000.0,   # 1000 minutes - effectively disabled
            mean_driver_inter_arrival_time=1000.0,  # 1000 minutes - effectively disabled
            
            # Pairing configuration - test with pairing enabled
            pairing_enabled=True,
            restaurants_proximity_threshold=3.0,
            customers_proximity_threshold=4.0,
            
            # Service duration - short for fast testing
            mean_service_duration=60,     # 1 hour
            service_duration_std_dev=15,  # 15 minutes std dev
            min_service_duration=30,      # 30 minutes minimum
            max_service_duration=120,     # 2 hours maximum
            
            # Assignment parameters - favorable for quick assignments
            throughput_factor=1.0,
            age_factor=0.1,
            immediate_assignment_threshold=10.0,  # High threshold for immediate assignment
            periodic_interval=5.0                 # 5 minutes for periodic optimization
        )
        
        # Experiment configuration - short duration, fixed seed
        experiment_config = ExperimentConfig(
            simulation_duration=30.0,        # 30 minutes - enough for complete workflows
            warmup_period=0.0,               # No warmup needed for controlled tests
            num_replications=1,              # Single replication for deterministic testing
            master_seed=42,                  # Fixed seed for reproducibility
            metrics_collection_interval=5.0, # Regular metrics collection
            event_recording_enabled=True     # Enable detailed event recording
        )
        
        return SimulationConfig(
            structural_config=structural_config,
            operational_config=operational_config,
            experiment_config=experiment_config
        )
    
    def test_single_order_complete_lifecycle(self, controlled_config):
        """
        Test the complete lifecycle of a single order from creation to delivery.
        
        This is the most basic end-to-end test that verifies the fundamental
        workflow works correctly: Order Creation → Assignment → Delivery → Completion.
        
        Workflow being tested:
        1. Order starts in CREATED state
        2. Assignment service assigns it to available driver
        3. Driver picks up the order (PICKED_UP state)
        4. Driver delivers the order (DELIVERED state)
        5. Driver becomes available again
        """
        # ARRANGE - Set up the controlled scenario
        
        # Create and initialize the simulation
        runner = SimulationRunner(controlled_config)
        runner.initialize()
        
        # Get access to repositories and services for manual entity creation
        env = runner.env
        order_repo = runner.repositories['order']
        driver_repo = runner.repositories['driver']
        delivery_unit_repo = runner.repositories['delivery_unit']
        
        # Create a test order manually (bypassing OrderArrivalService)
        test_order = Order(
            order_id="TEST_O1",
            restaurant_location=[3.0, 3.0],    # Known restaurant location
            customer_location=[5.0, 5.0],      # 2.83 km from restaurant
            arrival_time=env.now
        )
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        # Create a test driver manually (bypassing DriverArrivalService)
        test_driver = Driver(
            driver_id="TEST_D1",
            initial_location=[3.0, 3.0],       # At the restaurant for fast pickup
            login_time=env.now,
            service_duration=60                # 1 hour service duration
        )
        test_driver.entity_type = EntityType.DRIVER
        driver_repo.add(test_driver)
        
        # Verify initial states
        assert test_order.state == OrderState.CREATED, "Order should start in CREATED state"
        assert test_driver.state == DriverState.AVAILABLE, "Driver should start AVAILABLE"
        
        # ACT - Run the simulation to let the workflow execute
        
        # Calculate expected delivery time for timing control
        # Distance from driver to restaurant: 0 km (already there)
        # Distance from restaurant to customer: 2.83 km
        # With driver speed of 2.0 km/min, delivery should take ~1.4 minutes
        # Add buffer for assignment processing time
        expected_delivery_time = 3.0
        
        # Run simulation long enough for complete delivery workflow
        runner.env.run(until=expected_delivery_time)
        
        # ASSERT - Verify the complete workflow executed correctly
        
        # Verify final order state progression
        assert test_order.state == OrderState.DELIVERED, "Order should be delivered"
        
        # Verify order timing progression is logical
        assert test_order.assignment_time is not None, "Order should have assignment time"
        assert test_order.pickup_time is not None, "Order should have pickup time"
        assert test_order.delivery_time is not None, "Order should have delivery time"
        
        # Verify timing sequence is correct
        assert test_order.arrival_time <= test_order.assignment_time, "Assignment should be after arrival"
        assert test_order.assignment_time <= test_order.pickup_time, "Pickup should be after assignment"
        assert test_order.pickup_time <= test_order.delivery_time, "Delivery should be after pickup"
        
        # Verify driver state progression
        assert test_driver.state == DriverState.AVAILABLE, "Driver should be available again after delivery"
        
        # Verify delivery unit was created and completed
        delivery_units = delivery_unit_repo.find_all()
        assert len(delivery_units) == 1, "Should create exactly one delivery unit"
        
        delivery_unit = delivery_units[0]
        assert delivery_unit.state == DeliveryUnitState.COMPLETED, "Delivery unit should be completed"
        assert delivery_unit.delivery_entity is test_order, "Delivery unit should reference the order"
        assert delivery_unit.driver is test_driver, "Delivery unit should reference the driver"
        
        # Verify driver completed deliveries list
        assert len(test_driver.completed_deliveries) == 1, "Driver should have one completed delivery"
        assert test_driver.completed_deliveries[0] is delivery_unit, "Driver should reference the completed unit"
    
    def test_pair_complete_lifecycle(self, controlled_config):
        """
        Test the complete lifecycle of a paired order delivery.
        
        This test verifies the more complex workflow where two orders are paired
        together and delivered as a unit. It tests:
        1. Two compatible orders get paired
        2. The pair gets assigned to a driver
        3. Both orders are picked up and delivered in sequence
        4. All entities reach correct final states
        """
        # ARRANGE - Set up the paired delivery scenario
        
        runner = SimulationRunner(controlled_config)
        runner.initialize()
        
        env = runner.env
        order_repo = runner.repositories['order']
        driver_repo = runner.repositories['driver']
        pair_repo = runner.repositories['pair']
        delivery_unit_repo = runner.repositories['delivery_unit']
        
        # Create two compatible orders for pairing
        order1 = Order(
            order_id="TEST_O1",
            restaurant_location=[4.0, 4.0],    # Same restaurant
            customer_location=[6.0, 6.0],      # 2.83 km from restaurant
            arrival_time=env.now
        )
        order1.entity_type = EntityType.ORDER
        order_repo.add(order1)
        
        order2 = Order(
            order_id="TEST_O2", 
            restaurant_location=[4.0, 4.0],    # Same restaurant as order1
            customer_location=[7.0, 7.0],      # 4.24 km from restaurant, close to order1 customer
            arrival_time=env.now + 0.1         # Slightly later arrival
        )
        order2.entity_type = EntityType.ORDER
        order_repo.add(order2)
        
        # Verify these orders meet pairing criteria
        restaurant_distance = calculate_distance(order1.restaurant_location, order2.restaurant_location)
        customer_distance = calculate_distance(order1.customer_location, order2.customer_location)
        
        assert restaurant_distance <= controlled_config.operational_config.restaurants_proximity_threshold, \
            "Orders should meet restaurant proximity criteria"
        assert customer_distance <= controlled_config.operational_config.customers_proximity_threshold, \
            "Orders should meet customer proximity criteria"
        
        # Create a driver positioned for efficient delivery
        test_driver = Driver(
            driver_id="TEST_D1",
            initial_location=[4.0, 4.0],       # At the shared restaurant
            login_time=env.now,
            service_duration=90                # 1.5 hours service duration
        )
        test_driver.entity_type = EntityType.DRIVER
        driver_repo.add(test_driver)
        
        # Trigger pairing by dispatching OrderCreatedEvent for the second order
        # (The first order is already in the system waiting for a match)
        from delivery_sim.events.order_events import OrderCreatedEvent
        runner.event_dispatcher.dispatch(OrderCreatedEvent(
            timestamp=env.now,
            order_id=order2.order_id,
            restaurant_id="R1",
            restaurant_location=order2.restaurant_location,
            customer_location=order2.customer_location
        ))
        
        # ACT - Run simulation for pairing and complete delivery
        
        # Allow time for pairing to occur first
        env.run(until=1.0)
        
        # Verify pairing occurred
        pairs = pair_repo.find_all()
        assert len(pairs) == 1, "Should create exactly one pair"
        
        created_pair = pairs[0]
        assert order1.state == OrderState.PAIRED, "Order 1 should be paired"
        assert order2.state == OrderState.PAIRED, "Order 2 should be paired"
        
        # Calculate expected delivery time for the pair
        # The pair should have an optimal sequence calculated
        # With driver speed 2.0 km/min and estimated total distance ~7-8 km,
        # delivery should complete within 6-8 minutes
        total_delivery_time = 10.0
        
        # Run simulation for complete pair delivery
        env.run(until=total_delivery_time)
        
        # ASSERT - Verify complete pair workflow
        
        # Verify both orders were delivered
        assert order1.state == OrderState.DELIVERED, "Order 1 should be delivered"
        assert order2.state == OrderState.DELIVERED, "Order 2 should be delivered"
        
        # Verify pair state progression
        assert created_pair.state == PairState.COMPLETED, "Pair should be completed"
        
        # Verify delivery unit was created for the pair
        delivery_units = delivery_unit_repo.find_all()
        assert len(delivery_units) == 1, "Should create exactly one delivery unit for the pair"
        
        delivery_unit = delivery_units[0]
        assert delivery_unit.delivery_entity is created_pair, "Delivery unit should reference the pair"
        assert delivery_unit.state == DeliveryUnitState.COMPLETED, "Delivery unit should be completed"
        
        # Verify driver completed the delivery
        assert test_driver.state == DriverState.AVAILABLE, "Driver should be available after completing pair delivery"
        assert len(test_driver.completed_deliveries) == 1, "Driver should have one completed delivery"
        
        # Verify pair tracking shows both orders were processed
        assert len(created_pair.delivered_orders) == 2, "Pair should track both delivered orders"
        assert order1.order_id in created_pair.delivered_orders, "Pair should track order 1 delivery"
        assert order2.order_id in created_pair.delivered_orders, "Pair should track order 2 delivery"
    
    def test_multiple_orders_assignment_priority(self, controlled_config):
        """
        Test assignment priority when multiple orders compete for a single driver.
        
        This test verifies that the assignment algorithm correctly prioritizes
        orders based on the cost function (distance, age, etc.) when resources
        are limited. It creates a scenario where one driver must choose between
        multiple waiting orders.
        """
        # ARRANGE - Create scenario with multiple orders and one driver
        
        runner = SimulationRunner(controlled_config)
        runner.initialize()
        
        env = runner.env
        order_repo = runner.repositories['order']
        driver_repo = runner.repositories['driver']
        delivery_unit_repo = runner.repositories['delivery_unit']
        
        # Create multiple orders with different characteristics
        
        # Order 1: Close to driver, recently arrived
        order1 = Order(
            order_id="TEST_O1",
            restaurant_location=[2.0, 2.0],
            customer_location=[3.0, 3.0],
            arrival_time=env.now - 1.0          # 1 minute old
        )
        order1.entity_type = EntityType.ORDER
        order_repo.add(order1)
        
        # Order 2: Far from driver, but older (should get age bonus)
        order2 = Order(
            order_id="TEST_O2",
            restaurant_location=[8.0, 8.0],
            customer_location=[9.0, 9.0],
            arrival_time=env.now - 15.0         # 15 minutes old (significant age factor)
        )
        order2.entity_type = EntityType.ORDER
        order_repo.add(order2)
        
        # Order 3: Medium distance, medium age
        order3 = Order(
            order_id="TEST_O3",
            restaurant_location=[5.0, 5.0],
            customer_location=[6.0, 6.0],
            arrival_time=env.now - 5.0          # 5 minutes old
        )
        order3.entity_type = EntityType.ORDER
        order_repo.add(order3)
        
        # Create single driver who will need to choose
        test_driver = Driver(
            driver_id="TEST_D1",
            initial_location=[1.0, 1.0],       # Closest to order1
            login_time=env.now,
            service_duration=120
        )
        test_driver.entity_type = EntityType.DRIVER
        driver_repo.add(test_driver)
        
        # ACT - Run simulation to trigger assignment decision
        
        # Since immediate assignment threshold is high (10.0), the assignment
        # should happen immediately when the driver becomes available
        env.run(until=2.0)
        
        # ASSERT - Verify the assignment decision was optimal
        
        # Only one order should be assigned (one driver available)
        assigned_orders = [o for o in [order1, order2, order3] if o.state == OrderState.ASSIGNED or o.state == OrderState.DELIVERED]
        unassigned_orders = [o for o in [order1, order2, order3] if o.state == OrderState.CREATED]
        
        assert len(assigned_orders) == 1, "Exactly one order should be assigned to the single driver"
        assert len(unassigned_orders) == 2, "Two orders should remain unassigned"
        
        # Verify that the assignment decision was based on the cost function
        # With age_factor=0.1 and throughput_factor=1.0:
        # Order1: base_cost ~= 1.4 + 1.4 = 2.8, age_discount = 0.1*1 = 0.1, adjusted = 2.7
        # Order2: base_cost ~= 10.0 + 1.4 = 11.4, age_discount = 0.1*15 = 1.5, adjusted = 8.9  
        # Order3: base_cost ~= 5.7 + 1.4 = 7.1, age_discount = 0.1*5 = 0.5, adjusted = 5.6
        # Order1 should be selected (lowest adjusted cost)
        
        assigned_order = assigned_orders[0]
        assert assigned_order.order_id == "TEST_O1", "Order 1 should be selected (lowest adjusted cost)"
        
        # Verify delivery unit was created correctly
        delivery_units = delivery_unit_repo.find_all()
        assert len(delivery_units) == 1, "Should create one delivery unit"
        
        delivery_unit = delivery_units[0]
        assert delivery_unit.assignment_path == "immediate", "Should be immediate assignment"
        assert delivery_unit.assignment_costs is not None, "Cost components should be recorded"
    
    def test_driver_logout_workflow(self, controlled_config):
        """
        Test the complete driver lifecycle including scheduled logout.
        
        This test verifies that drivers properly log out when their intended
        service time expires, and that the logout doesn't interfere with
        ongoing deliveries. It tests:
        1. Driver logs in and gets scheduled for logout
        2. Driver completes deliveries during service period
        3. Driver logs out at intended time (if available)
        4. Driver logout is deferred if busy at logout time
        """
        # ARRANGE - Create scenario with driver that will reach logout time
        
        # Modify config for shorter service duration to test logout
        structural_config = controlled_config.structural_config
        operational_config = controlled_config.operational_config
        experiment_config = controlled_config.experiment_config
        
        # Override service duration for shorter testing
        operational_config.mean_service_duration = 10  # 10 minutes service
        operational_config.min_service_duration = 10
        operational_config.max_service_duration = 10
        
        modified_config = SimulationConfig(structural_config, operational_config, experiment_config)
        
        runner = SimulationRunner(modified_config)
        runner.initialize()
        
        env = runner.env
        order_repo = runner.repositories['order']
        driver_repo = runner.repositories['driver']
        
        # Create a driver with 10-minute service duration
        test_driver = Driver(
            driver_id="TEST_D1",
            initial_location=[3.0, 3.0],
            login_time=env.now,
            service_duration=10                # Will logout at t=10
        )
        test_driver.entity_type = EntityType.DRIVER
        driver_repo.add(test_driver)
        
        # Dispatch driver login event to trigger logout scheduling
        from delivery_sim.events.driver_events import DriverLoggedInEvent
        runner.event_dispatcher.dispatch(DriverLoggedInEvent(
            timestamp=env.now,
            driver_id=test_driver.driver_id,
            initial_location=test_driver.location,
            service_duration=test_driver.service_duration
        ))
        
        # Create an order for early delivery (should complete before logout)
        early_order = Order(
            order_id="TEST_O1",
            restaurant_location=[3.0, 3.0],
            customer_location=[4.0, 4.0],      # Close for quick delivery
            arrival_time=env.now + 1.0
        )
        early_order.entity_type = EntityType.ORDER
        order_repo.add(early_order)
        
        # ACT & ASSERT - Test the complete driver lifecycle
        
        # Phase 1: Early delivery (before logout time)
        env.run(until=5.0)  # Run until t=5 (before logout at t=10)
        
        # Verify early order was delivered and driver is available
        assert early_order.state == OrderState.DELIVERED, "Early order should be delivered"
        assert test_driver.state == DriverState.AVAILABLE, "Driver should be available after early delivery"
        
        # Phase 2: Reach logout time while driver is available
        env.run(until=11.0)  # Run past logout time (t=10)
        
        # Verify driver logged out
        assert test_driver.state == DriverState.OFFLINE, "Driver should log out at intended time"
        assert test_driver.actual_logout_time is not None, "Driver should have actual logout time recorded"
        
        # Verify logout time is reasonable (should be close to intended logout time)
        expected_logout_time = test_driver.login_time + test_driver.service_duration
        assert abs(test_driver.actual_logout_time - expected_logout_time) < 1.0, \
            "Actual logout time should be close to intended logout time"
    
    def test_workflow_with_periodic_assignment(self, controlled_config):
        """
        Test workflow where entities are assigned through periodic optimization.
        
        This test verifies that the periodic assignment process correctly
        handles entities that weren't assigned immediately. It creates a
        scenario where immediate assignment fails but periodic assignment succeeds.
        """
        # ARRANGE - Create scenario that will trigger periodic assignment
        
        # Modify config to have low immediate assignment threshold
        # but keep periodic assignment enabled
        controlled_config.operational_config.immediate_assignment_threshold = 1.0  # Very restrictive
        controlled_config.operational_config.periodic_interval = 3.0              # Every 3 minutes
        
        runner = SimulationRunner(controlled_config)
        runner.initialize()
        
        env = runner.env
        order_repo = runner.repositories['order']
        driver_repo = runner.repositories['driver']
        delivery_unit_repo = runner.repositories['delivery_unit']
        
        # Create an order and driver with high immediate assignment cost
        # but reasonable overall cost
        test_order = Order(
            order_id="TEST_O1",
            restaurant_location=[1.0, 1.0],
            customer_location=[2.0, 2.0],
            arrival_time=env.now
        )
        test_order.entity_type = EntityType.ORDER
        order_repo.add(test_order)
        
        # Driver far enough to exceed immediate threshold but reasonable for periodic
        test_driver = Driver(
            driver_id="TEST_D1",
            initial_location=[5.0, 5.0],       # Far enough to exceed immediate threshold
            login_time=env.now,
            service_duration=60
        )
        test_driver.entity_type = EntityType.DRIVER
        driver_repo.add(test_driver)
        
        # ACT - Run simulation through periodic assignment cycle
        
        # Phase 1: Immediate assignment should fail
        env.run(until=1.0)
        assert test_order.state == OrderState.CREATED, "Order should remain unassigned after immediate attempt"
        assert test_driver.state == DriverState.AVAILABLE, "Driver should remain available after immediate attempt"
        assert len(delivery_unit_repo.find_all()) == 0, "No delivery unit should be created yet"
        
        # Phase 2: Wait for periodic assignment (occurs at t=3.0)
        env.run(until=4.0)
        
        # ASSERT - Verify periodic assignment succeeded
        assert test_order.state == OrderState.ASSIGNED, "Order should be assigned through periodic optimization"
        assert test_driver.state == DriverState.DELIVERING, "Driver should be delivering after periodic assignment"
        
        delivery_units = delivery_unit_repo.find_all()
        assert len(delivery_units) == 1, "Should create delivery unit through periodic assignment"
        
        delivery_unit = delivery_units[0]
        assert delivery_unit.assignment_path == "periodic", "Should be marked as periodic assignment"
        
        # Phase 3: Complete the delivery workflow
        env.run(until=10.0)  # Allow time for delivery completion
        
        assert test_order.state == OrderState.DELIVERED, "Order should be delivered"
        assert test_driver.state == DriverState.AVAILABLE, "Driver should be available after delivery"
        assert delivery_unit.state == DeliveryUnitState.COMPLETED, "Delivery unit should be completed"


