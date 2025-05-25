# tests/integration/test_periodic_assignment_optimization.py
"""
Integration tests for Area 4: Periodic Assignment Batch Optimization

This test file verifies the batch optimization process that handles assignment
decisions for entities that weren't successfully handled through immediate assignment.
The periodic assignment represents sophisticated algorithmic coordination in the system,
using the Hungarian algorithm to find globally optimal assignment patterns.

Key concepts being tested:
- The SimPy process that triggers periodic optimization at regular intervals
- Collection of all waiting entities and available drivers for batch processing
- Integration with the Hungarian algorithm for optimal matching
- Handling of resource imbalances (more drivers than orders or vice versa)
- Simultaneous creation of multiple assignments with proper state management
- Consistency between immediate and periodic assignment cost calculations

Note: The cost matrix generation logic itself is tested at the unit level.
These integration tests focus on how the periodic process coordinates all components.
"""

import pytest
import simpy
import numpy as np
from unittest.mock import Mock, patch

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
from delivery_sim.entities.states import OrderState, DriverState, PairState
from delivery_sim.utils.entity_type_utils import EntityType


class TestPeriodicAssignmentOptimization:
    """
    Test suite for the periodic batch assignment optimization process.
    
    These tests verify that the system correctly:
    1. Triggers periodic optimization at configured intervals
    2. Collects all eligible entities for batch processing
    3. Produces optimal assignment solutions using the Hungarian algorithm
    4. Handles various resource balance scenarios
    5. Creates multiple assignments atomically with proper state management
    """
    
    @pytest.fixture
    def test_config(self):
        """Configuration for periodic assignment testing."""
        class TestConfig:
            def __init__(self):
                # Assignment parameters
                self.immediate_assignment_threshold = 5.0  # Not used in periodic
                self.periodic_interval = 5.0  # Run optimization every 5 minutes
                self.throughput_factor = 1.0  # Same cost calculation as immediate
                self.age_factor = 0.1  # Same cost calculation as immediate
                self.driver_speed = 0.5
                
                # Pairing configuration
                self.pairing_enabled = True  # Test with pairs to verify complex scenarios
        
        return TestConfig()
    
    @pytest.fixture
    def test_environment(self):
        """Set up the test environment with all necessary components."""
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
    
    # ===== Test 1: Periodic Process Timing =====
    
    def test_periodic_process_runs_at_configured_intervals(self, test_environment, test_config):
        """
        Test that the periodic assignment process runs at the configured interval.
        
        This verifies the SimPy process integration - that the system correctly
        schedules and executes periodic optimization at regular time intervals.
        This is like setting up a cron job that runs every N minutes.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
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
        
        # Track when periodic assignment runs
        periodic_runs = []
        original_method = assignment_service.perform_periodic_assignment
        
        def track_periodic_runs(epoch_count=0):
            # Record the simulation time when this runs
            periodic_runs.append({
                'time': env.now,
                'epoch': epoch_count
            })
            # Don't actually run the optimization (no entities to assign)
            return original_method(epoch_count)
        
        assignment_service.perform_periodic_assignment = track_periodic_runs
        
        # ACT - Run simulation for multiple intervals
        # With interval of 5 minutes, running for 22 minutes should trigger
        # at times: 5, 10, 15, 20 (4 times total)
        env.run(until=22)
        
        # ASSERT
        assert len(periodic_runs) == 4, f"Should run 4 times in 22 minutes with 5-minute intervals, but ran {len(periodic_runs)} times"
        
        # Verify the timing of each run
        expected_times = [5.0, 10.0, 15.0, 20.0]
        for i, run in enumerate(periodic_runs):
            assert run['time'] == expected_times[i], f"Run {i+1} should occur at time {expected_times[i]}, but occurred at {run['time']}"
            assert run['epoch'] == i + 1, f"Epoch count should increment correctly"
    
    # ===== Test 2: Entity Collection =====
    
    def test_periodic_collects_all_waiting_entities_and_drivers(self, test_environment, test_config):
        """
        Test that periodic assignment correctly identifies all entities needing assignment.
        
        This verifies that the system properly collects:
        - Unassigned single orders (state = CREATED)
        - Unassigned pairs (state = CREATED)
        - Available drivers (state = AVAILABLE)
        
        It's like taking a complete inventory before running the optimization.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
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
        
        # Create test entities in various states
        
        # Unassigned orders (should be collected)
        order1 = Order("O1", [1, 1], [2, 2], env.now)
        order1.entity_type = EntityType.ORDER
        order1.state = OrderState.CREATED
        order_repo.add(order1)
        
        # Assigned order (should NOT be collected)
        order2 = Order("O2", [3, 3], [4, 4], env.now)
        order2.entity_type = EntityType.ORDER
        order2.state = OrderState.ASSIGNED
        order_repo.add(order2)
        
        # Unassigned pair (should be collected)
        order3 = Order("O3", [5, 5], [6, 6], env.now)
        order4 = Order("O4", [5, 5], [7, 7], env.now)
        pair1 = Pair(order3, order4, env.now)
        pair1.entity_type = EntityType.PAIR
        pair1.state = PairState.CREATED
        pair_repo.add(pair1)
        
        # Available drivers (should be collected)
        driver1 = Driver("D1", [2, 2], env.now, 120)
        driver1.entity_type = EntityType.DRIVER
        driver1.state = DriverState.AVAILABLE
        driver_repo.add(driver1)
        
        driver2 = Driver("D2", [6, 6], env.now, 120)
        driver2.entity_type = EntityType.DRIVER
        driver2.state = DriverState.AVAILABLE
        driver_repo.add(driver2)
        
        # Busy driver (should NOT be collected)
        driver3 = Driver("D3", [8, 8], env.now, 120)
        driver3.entity_type = EntityType.DRIVER
        driver3.state = DriverState.DELIVERING
        driver_repo.add(driver3)
        
        # Track what entities are collected for optimization
        collected_data = {}
        
        # Patch the Hungarian algorithm to capture what's being optimized
        with patch('delivery_sim.services.assignment_service.linear_sum_assignment') as mock_hungarian:
            # Make it return a simple assignment (entity 0 → driver 0, entity 1 → driver 1)
            mock_hungarian.return_value = (np.array([0, 1]), np.array([0, 1]))
            
            # Override _generate_cost_matrix to capture the entities
            original_matrix_method = assignment_service._generate_cost_matrix
            
            def capture_entities(waiting_entities, available_drivers):
                collected_data['waiting_entities'] = waiting_entities[:]
                collected_data['available_drivers'] = available_drivers[:]
                # Return a simple matrix for the test
                return [[1.0] * len(available_drivers) for _ in waiting_entities]
            
            assignment_service._generate_cost_matrix = capture_entities
            
            # ACT - Manually trigger periodic assignment
            assignment_service.perform_periodic_assignment()
        
        # ASSERT
        # Verify correct entities were collected
        assert len(collected_data['waiting_entities']) == 2, "Should collect 1 order + 1 pair"
        assert len(collected_data['available_drivers']) == 2, "Should collect 2 available drivers"
        
        # Verify specific entities
        entity_ids = [e.order_id if hasattr(e, 'order_id') else e.pair_id 
                      for e in collected_data['waiting_entities']]
        assert "O1" in entity_ids, "Unassigned order O1 should be collected"
        assert pair1.pair_id in entity_ids, "Unassigned pair should be collected"
        assert "O2" not in entity_ids, "Assigned order O2 should NOT be collected"
        
        driver_ids = [d.driver_id for d in collected_data['available_drivers']]
        assert "D1" in driver_ids and "D2" in driver_ids, "Available drivers should be collected"
        assert "D3" not in driver_ids, "Busy driver D3 should NOT be collected"
    
    # ===== Test 3: Balanced Assignment Scenario =====
    
    def test_periodic_creates_optimal_assignments_balanced_scenario(self, test_environment, test_config):
        """
        Test periodic assignment with equal numbers of entities and drivers.
        
        This is the "ideal" scenario where we have exactly the right number of
        drivers for the waiting deliveries. The Hungarian algorithm should find
        the optimal assignment that minimizes total system cost.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
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
        
        # Create a scenario with clear optimal assignments
        # Order at [1,1] should match with Driver at [1,1] (distance = 0)
        # Order at [5,5] should match with Driver at [5,5] (distance = 0)
        
        order1 = Order("O1", [1, 1], [2, 2], env.now - 10)  # 10 minutes old
        order1.entity_type = EntityType.ORDER
        order_repo.add(order1)
        
        order2 = Order("O2", [5, 5], [6, 6], env.now - 5)   # 5 minutes old
        order2.entity_type = EntityType.ORDER
        order_repo.add(order2)
        
        driver1 = Driver("D1", [1, 1], env.now, 120)  # At order1's restaurant
        driver1.entity_type = EntityType.DRIVER
        driver1.state = DriverState.AVAILABLE
        driver_repo.add(driver1)
        
        driver2 = Driver("D2", [5, 5], env.now, 120)  # At order2's restaurant
        driver2.entity_type = EntityType.DRIVER
        driver2.state = DriverState.AVAILABLE
        driver_repo.add(driver2)
        
        # Track assignment events
        assignment_events = []
        event_dispatcher.register(DeliveryUnitAssignedEvent, lambda e: assignment_events.append(e))
        
        # ACT - Run periodic assignment
        assignment_service.perform_periodic_assignment()
        
        # ASSERT
        # Should create exactly 2 assignments
        assert len(assignment_events) == 2, "Should create 2 assignments in balanced scenario"
        
        # Verify optimal matching occurred
        assignments = {}
        for event in assignment_events:
            assignments[event.entity_id] = event.driver_id
        
        assert assignments.get("O1") == "D1", "Order O1 should be assigned to closest driver D1"
        assert assignments.get("O2") == "D2", "Order O2 should be assigned to closest driver D2"
        
        # Verify all entities are now assigned
        assert order1.state == OrderState.ASSIGNED, "Order O1 should be assigned"
        assert order2.state == OrderState.ASSIGNED, "Order O2 should be assigned"
        assert driver1.state == DriverState.DELIVERING, "Driver D1 should be delivering"
        assert driver2.state == DriverState.DELIVERING, "Driver D2 should be delivering"
        
        # Verify delivery units were created
        delivery_units = delivery_unit_repo.find_all()
        assert len(delivery_units) == 2, "Should create 2 delivery units"
        
        # Verify assignment path is marked as "periodic"
        for unit in delivery_units:
            assert unit.assignment_path == "periodic", "Should be marked as periodic assignment"
    
    # ===== Test 4: More Entities Than Drivers =====
    
    def test_periodic_handles_more_entities_than_drivers(self, test_environment, test_config):
        """
        Test periodic assignment when there are more orders/pairs than available drivers.
        
        This tests the system's behavior under high demand. The Hungarian algorithm
        should assign drivers to minimize total cost, leaving some entities unassigned.
        This scenario helps verify that the system gracefully handles resource shortage.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
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
        
        # Create 4 orders but only 2 drivers
        orders = []
        for i in range(4):
            order = Order(f"O{i+1}", [i*2, i*2], [i*2+1, i*2+1], env.now - (10-i))
            order.entity_type = EntityType.ORDER
            order_repo.add(order)
            orders.append(order)
        
        drivers = []
        for i in range(2):
            driver = Driver(f"D{i+1}", [i*4, i*4], env.now, 120)
            driver.entity_type = EntityType.DRIVER
            driver.state = DriverState.AVAILABLE
            driver_repo.add(driver)
            drivers.append(driver)
        
        # Track assignment events
        assignment_events = []
        event_dispatcher.register(DeliveryUnitAssignedEvent, lambda e: assignment_events.append(e))
        
        # ACT - Run periodic assignment
        assignment_service.perform_periodic_assignment()
        
        # ASSERT
        # Should create exactly 2 assignments (limited by driver count)
        assert len(assignment_events) == 2, "Should create 2 assignments (limited by drivers)"
        
        # Count how many orders got assigned
        assigned_orders = [o for o in orders if o.state == OrderState.ASSIGNED]
        unassigned_orders = [o for o in orders if o.state == OrderState.CREATED]
        
        assert len(assigned_orders) == 2, "Exactly 2 orders should be assigned"
        assert len(unassigned_orders) == 2, "Exactly 2 orders should remain unassigned"
        
        # All drivers should be busy
        for driver in drivers:
            assert driver.state == DriverState.DELIVERING, f"Driver {driver.driver_id} should be delivering"
        
        # The oldest orders should generally be prioritized due to age factor
        # (though exact assignment depends on distances too)
        assert any(o.order_id in ["O1", "O2"] for o in assigned_orders), \
            "Older orders should generally be prioritized"
    
    # ===== Test 5: More Drivers Than Entities =====
    
    def test_periodic_handles_more_drivers_than_entities(self, test_environment, test_config):
        """
        Test periodic assignment when there are more drivers than orders/pairs.
        
        This tests the system under low demand. Some drivers will remain idle,
        and the system should choose the best-positioned drivers for the available work.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
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
        
        # Create 2 orders but 4 drivers
        order1 = Order("O1", [2, 2], [3, 3], env.now - 5)
        order1.entity_type = EntityType.ORDER
        order_repo.add(order1)
        
        order2 = Order("O2", [6, 6], [7, 7], env.now - 3)
        order2.entity_type = EntityType.ORDER
        order_repo.add(order2)
        
        # Create drivers at various distances
        drivers = [
            Driver("D1", [2.1, 2.1], env.now, 120),  # Very close to order1
            Driver("D2", [5.9, 5.9], env.now, 120),  # Very close to order2
            Driver("D3", [9, 9], env.now, 120),      # Far from both
            Driver("D4", [10, 10], env.now, 120),    # Even farther
        ]
        
        for driver in drivers:
            driver.entity_type = EntityType.DRIVER
            driver.state = DriverState.AVAILABLE
            driver_repo.add(driver)
        
        # Track assignment events
        assignment_events = []
        event_dispatcher.register(DeliveryUnitAssignedEvent, lambda e: assignment_events.append(e))
        
        # ACT - Run periodic assignment
        assignment_service.perform_periodic_assignment()
        
        # ASSERT
        # Should create exactly 2 assignments (limited by order count)
        assert len(assignment_events) == 2, "Should create 2 assignments (limited by orders)"
        
        # The closest drivers should be selected
        assigned_driver_ids = [e.driver_id for e in assignment_events]
        assert "D1" in assigned_driver_ids, "Closest driver D1 should be assigned"
        assert "D2" in assigned_driver_ids, "Closest driver D2 should be assigned"
        
        # Verify the far drivers remain available
        assert drivers[2].state == DriverState.AVAILABLE, "Far driver D3 should remain available"
        assert drivers[3].state == DriverState.AVAILABLE, "Far driver D4 should remain available"
    
    # ===== Test 6: Mixed Entity Types (Orders and Pairs) =====
    
    def test_periodic_handles_mixed_orders_and_pairs(self, test_environment, test_config):
        """
        Test that periodic assignment correctly handles a mix of single orders and pairs.
        
        This verifies that the cost calculation properly accounts for the different
        characteristics of orders vs pairs (pairs deliver 2 orders, have different
        routing, etc.) and that the optimal assignment considers these differences.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
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
        
        # Create a single order
        single_order = Order("O1", [1, 1], [2, 2], env.now - 5)
        single_order.entity_type = EntityType.ORDER
        order_repo.add(single_order)
        
        # Create a pair (represents 2 orders being delivered together)
        order2 = Order("O2", [5, 5], [6, 6], env.now - 4)
        order3 = Order("O3", [5, 5], [6.5, 6.5], env.now - 3)
        pair = Pair(order2, order3, env.now - 3)
        pair.entity_type = EntityType.PAIR
        pair.state = PairState.CREATED
        pair.optimal_sequence = [[5, 5], [6, 6], [6.5, 6.5]]
        pair.optimal_cost = 1.5  # Short delivery route
        pair_repo.add(pair)
        
        # Create drivers
        driver1 = Driver("D1", [1, 1], env.now, 120)  # Close to single order
        driver1.entity_type = EntityType.DRIVER
        driver1.state = DriverState.AVAILABLE
        driver_repo.add(driver1)
        
        driver2 = Driver("D2", [5, 5], env.now, 120)  # Close to pair
        driver2.entity_type = EntityType.DRIVER
        driver2.state = DriverState.AVAILABLE
        driver_repo.add(driver2)
        
        # Track assignment events
        assignment_events = []
        event_dispatcher.register(DeliveryUnitAssignedEvent, lambda e: assignment_events.append(e))
        
        # ACT - Run periodic assignment
        assignment_service.perform_periodic_assignment()
        
        # ASSERT
        # Should create 2 assignments
        assert len(assignment_events) == 2, "Should create assignments for both entities"
        
        # Verify both entity types were handled
        entity_types_assigned = {e.entity_type for e in assignment_events}
        assert EntityType.ORDER in entity_types_assigned, "Single order should be assigned"
        assert EntityType.PAIR in entity_types_assigned, "Pair should be assigned"
        
        # Verify the pair's constituent orders were updated
        assert order2.state == OrderState.ASSIGNED, "Order in pair should be assigned"
        assert order3.state == OrderState.ASSIGNED, "Order in pair should be assigned"
        
        # Verify delivery units reference the correct entity types
        delivery_units = delivery_unit_repo.find_all()
        entity_types_in_units = {unit.delivery_entity.entity_type for unit in delivery_units}
        assert EntityType.ORDER in entity_types_in_units
        assert EntityType.PAIR in entity_types_in_units
    
    # ===== Test 7: Cost Consistency =====
    
    def test_periodic_uses_same_cost_calculation_as_immediate(self, test_environment, test_config):
        """
        Test that periodic assignment uses the same cost calculation as immediate assignment.
        
        This is crucial for fairness - an assignment shouldn't have different costs
        depending on whether it goes through immediate or periodic assignment.
        The only difference should be the threshold check.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
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
        
        # Create a simple scenario
        order = Order("O1", [3, 3], [5, 5], env.now - 10)  # 10 minutes old
        order.entity_type = EntityType.ORDER
        order_repo.add(order)
        
        driver = Driver("D1", [1, 1], env.now, 120)
        driver.entity_type = EntityType.DRIVER
        driver.state = DriverState.AVAILABLE
        driver_repo.add(driver)
        
        # First, calculate cost using the immediate assignment method
        immediate_cost, immediate_components = assignment_service.calculate_adjusted_cost(driver, order)
        
        # Track what gets stored in delivery units during periodic assignment
        created_units = []
        
        original_add = delivery_unit_repo.add
        def track_units(unit):
            created_units.append(unit)
            return original_add(unit)
        
        delivery_unit_repo.add = track_units
        
        # ACT - Run periodic assignment
        assignment_service.perform_periodic_assignment()
        
        # ASSERT
        assert len(created_units) == 1, "Should create one delivery unit"
        
        unit = created_units[0]
        periodic_cost = unit.assignment_costs['adjusted_cost']
        
        # The costs should be identical
        assert abs(periodic_cost - immediate_cost) < 0.01, \
            f"Periodic cost ({periodic_cost}) should match immediate cost ({immediate_cost})"
        
        # Verify all cost components match
        assert unit.assignment_costs['base_cost'] == immediate_components['base_cost']
        assert unit.assignment_costs['throughput_discount'] == immediate_components['throughput_component']
        assert unit.assignment_costs['age_discount'] == immediate_components['age_discount']
    
    # ===== Test 8: No Entities or Drivers =====
    
    def test_periodic_handles_empty_scenarios_gracefully(self, test_environment, test_config):
        """
        Test that periodic assignment handles edge cases with no entities or drivers.
        
        The system should handle these scenarios without errors, simply skipping
        the optimization when there's nothing to optimize.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        driver_repo = test_environment["driver_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
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
        
        # Track if Hungarian algorithm is called
        hungarian_called = []
        
        with patch('delivery_sim.services.assignment_service.linear_sum_assignment') as mock_hungarian:
            mock_hungarian.side_effect = lambda x: hungarian_called.append(True) or ([], [])
            
            # Test 1: No entities, but drivers exist
            driver = Driver("D1", [1, 1], env.now, 120)
            driver.entity_type = EntityType.DRIVER
            driver.state = DriverState.AVAILABLE
            driver_repo.add(driver)
            
            # ACT
            assignment_service.perform_periodic_assignment()
            
            # ASSERT
            assert len(hungarian_called) == 0, "Hungarian algorithm should not run with no entities"
            
            # Test 2: No drivers, but entities exist
            driver_repo.drivers.clear()  # Remove the driver
            
            order = Order("O1", [1, 1], [2, 2], env.now)
            order.entity_type = EntityType.ORDER
            order_repo.add(order)
            
            # ACT
            assignment_service.perform_periodic_assignment()
            
            # ASSERT
            assert len(hungarian_called) == 0, "Hungarian algorithm should not run with no drivers"
            
            # Test 3: No entities and no drivers
            order_repo.orders.clear()
            
            # ACT - Should complete without error
            assignment_service.perform_periodic_assignment()
            
            # ASSERT
            assert len(hungarian_called) == 0, "Hungarian algorithm should not run with nothing to assign"