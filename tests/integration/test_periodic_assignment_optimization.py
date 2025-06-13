# tests/integration/test_periodic_assignment_optimization.py
"""
Integration tests for Periodic Assignment Global Optimization (Priority Scoring)

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

Updated for priority scoring system:
- AssignmentService now requires priority_scorer parameter
- Tests verify score-based optimization instead of cost-based
- Hungarian algorithm now maximizes scores instead of minimizing costs
- Assignment data stored as assignment_scores (instead of assignment_costs)
"""

import pytest
import simpy
import numpy as np
from unittest.mock import Mock, patch

# Import core components
from delivery_sim.events.event_dispatcher import EventDispatcher
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
from delivery_sim.utils.location_utils import calculate_distance


class TestPeriodicAssignmentOptimization:
    """
    Test suite for the periodic assignment optimization process using priority scoring.
    
    Updated to test score-based global optimization instead of cost minimization.
    """
    
    @pytest.fixture
    def test_config(self):
        """Configuration for periodic assignment testing."""
        class TestConfig:
            def __init__(self):
                self.immediate_assignment_threshold = 75.0
                self.periodic_interval = 3.0
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
        """Create a mock priority scorer for periodic assignment testing."""
        scorer = Mock()
        # Default return: reasonable score for testing
        scorer.calculate_priority_score.return_value = (80.0, {
            "distance_score": 0.8,
            "throughput_score": 0.5,
            "fairness_score": 0.9,
            "combined_score_0_1": 0.80,
            "total_distance": 7.0,
            "num_orders": 1,
            "wait_time_minutes": 4.0
        })
        return scorer
    
    # ===== Test 1: Entity Collection =====
    
    def test_periodic_collects_waiting_entities_and_available_drivers(self, test_environment, test_config, mock_priority_scorer):
        """
        Test that periodic assignment correctly collects waiting entities and available drivers.
        
        This verifies that the system properly takes inventory before running optimization.
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
        # Initialize optimal_sequence and optimal_cost for pair1
        pair1.optimal_sequence = [
            order3.restaurant_location, 
            order3.customer_location, 
            order4.customer_location
        ]
        pair1.optimal_cost = (
            calculate_distance(order3.restaurant_location, order3.customer_location) +
            calculate_distance(order3.customer_location, order4.customer_location)
        )
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
            
            # Override _generate_score_matrix to capture the entities
            original_matrix_method = assignment_service._generate_score_matrix
            
            def capture_entities(waiting_entities, available_drivers):
                collected_data['waiting_entities'] = waiting_entities[:]
                collected_data['available_drivers'] = available_drivers[:]
                # Return a simple matrix for the test
                return np.array([[80.0] * len(available_drivers) for _ in waiting_entities])
            
            assignment_service._generate_score_matrix = capture_entities
            
            # ACT - Manually trigger periodic assignment
            assignment_service.perform_periodic_assignment()
        
        # ASSERT
        # Verify correct entities were collected
        assert len(collected_data['waiting_entities']) == 2, "Should collect 2 waiting entities"
        assert len(collected_data['available_drivers']) == 2, "Should collect 2 available drivers"
        
        # Verify specific entities
        waiting_ids = [e.order_id if hasattr(e, 'order_id') else e.pair_id for e in collected_data['waiting_entities']]
        assert "O1" in waiting_ids, "Should collect unassigned order O1"
        assert pair1.pair_id in waiting_ids, "Should collect unassigned pair"
        assert "O2" not in waiting_ids, "Should NOT collect assigned order O2"
        
        driver_ids = [d.driver_id for d in collected_data['available_drivers']]
        assert "D1" in driver_ids, "Should collect available driver D1"
        assert "D2" in driver_ids, "Should collect available driver D2"
        assert "D3" not in driver_ids, "Should NOT collect busy driver D3"
    
    # ===== Test 2: Score Matrix Generation =====
    
    def test_periodic_generates_score_matrix_correctly(self, test_environment, test_config, mock_priority_scorer):
        """
        Test that periodic assignment generates the correct score matrix for optimization.
        
        This verifies the score matrix has proper dimensions and contains expected values.
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
        
        # Create 2 entities and 3 drivers for clear matrix testing
        order1 = Order("O1", [1, 1], [2, 2], env.now)
        order1.entity_type = EntityType.ORDER
        order1.state = OrderState.CREATED
        order_repo.add(order1)
        
        order2 = Order("O2", [3, 3], [4, 4], env.now)
        order2.entity_type = EntityType.ORDER
        order2.state = OrderState.CREATED
        order_repo.add(order2)
        
        driver1 = Driver("D1", [0, 0], env.now, 120)
        driver1.entity_type = EntityType.DRIVER
        driver1.state = DriverState.AVAILABLE
        driver_repo.add(driver1)
        
        driver2 = Driver("D2", [2, 2], env.now, 120)
        driver2.entity_type = EntityType.DRIVER
        driver2.state = DriverState.AVAILABLE
        driver_repo.add(driver2)
        
        driver3 = Driver("D3", [5, 5], env.now, 120)
        driver3.entity_type = EntityType.DRIVER
        driver3.state = DriverState.AVAILABLE
        driver_repo.add(driver3)
        
        # Set up priority scorer to return predictable scores
        # Order matters: O1-D1, O1-D2, O1-D3, O2-D1, O2-D2, O2-D3
        expected_scores = [65.0, 75.0, 55.0, 70.0, 80.0, 60.0]
        mock_priority_scorer.calculate_priority_score.side_effect = [
            (score, {"combined_score_0_1": score/100}) for score in expected_scores
        ]
        
        # Capture the generated matrix
        captured_matrix = None
        
        with patch('delivery_sim.services.assignment_service.linear_sum_assignment') as mock_hungarian:
            mock_hungarian.return_value = (np.array([0, 1]), np.array([0, 1]))
            
            # Override to capture matrix
            original_method = assignment_service._generate_score_matrix
            def capture_matrix(*args, **kwargs):
                nonlocal captured_matrix
                captured_matrix = original_method(*args, **kwargs)
                return captured_matrix
            
            assignment_service._generate_score_matrix = capture_matrix
            
            # ACT - Trigger periodic assignment
            assignment_service.perform_periodic_assignment()
        
        # ASSERT
        # Verify matrix dimensions
        assert captured_matrix is not None, "Score matrix should be generated"
        assert captured_matrix.shape == (2, 3), "Matrix should be 2 entities × 3 drivers"
        
        # Verify specific scores in matrix
        assert captured_matrix[0, 0] == 65.0, "O1-D1 score should be 65.0"
        assert captured_matrix[0, 1] == 75.0, "O1-D2 score should be 75.0"
        assert captured_matrix[0, 2] == 55.0, "O1-D3 score should be 55.0"
        assert captured_matrix[1, 0] == 70.0, "O2-D1 score should be 70.0"
        assert captured_matrix[1, 1] == 80.0, "O2-D2 score should be 80.0"
        assert captured_matrix[1, 2] == 60.0, "O2-D3 score should be 60.0"
        
        # Verify priority scorer was called correct number of times
        assert mock_priority_scorer.calculate_priority_score.call_count == 6, "Should calculate 6 scores (2×3)"
    
    # ===== Test 3: Hungarian Algorithm Integration =====
    
    def test_periodic_uses_hungarian_algorithm_for_score_maximization(self, test_environment, test_config, mock_priority_scorer):
        """
        Test that periodic assignment uses Hungarian algorithm to maximize scores.
        
        This verifies that the optimization correctly finds maximum score assignments.
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
        
        # Create simple scenario: 2 orders, 2 drivers
        order1 = Order("O1", [1, 1], [2, 2], env.now)
        order1.entity_type = EntityType.ORDER
        order1.state = OrderState.CREATED
        order_repo.add(order1)
        
        order2 = Order("O2", [4, 4], [5, 5], env.now)
        order2.entity_type = EntityType.ORDER
        order2.state = OrderState.CREATED
        order_repo.add(order2)
        
        driver1 = Driver("D1", [0, 0], env.now, 120)
        driver1.entity_type = EntityType.DRIVER
        driver1.state = DriverState.AVAILABLE
        driver_repo.add(driver1)
        
        driver2 = Driver("D2", [3, 3], env.now, 120)
        driver2.entity_type = EntityType.DRIVER
        driver2.state = DriverState.AVAILABLE
        driver_repo.add(driver2)
        
        # Set up scores where optimal assignment is: O1→D1 (90), O2→D2 (85)
        # vs suboptimal: O1→D2 (60), O2→D1 (70)
        score_matrix = np.array([
            [90.0, 60.0],  # Order1 to [Driver1, Driver2]
            [70.0, 85.0]   # Order2 to [Driver1, Driver2]
        ])
        
        mock_priority_scorer.calculate_priority_score.side_effect = [
            (90.0, {"combined_score_0_1": 0.90}),  # O1-D1
            (60.0, {"combined_score_0_1": 0.60}),  # O1-D2
            (70.0, {"combined_score_0_1": 0.70}),  # O2-D1
            (85.0, {"combined_score_0_1": 0.85})   # O2-D2
        ]
        
        # Track Hungarian algorithm calls
        hungarian_calls = []
        
        with patch('delivery_sim.services.assignment_service.linear_sum_assignment') as mock_hungarian:
            def track_hungarian_call(matrix, maximize=False):
                hungarian_calls.append({"matrix": matrix.copy(), "maximize": maximize})
                # Return optimal assignment: O1→D1 (row 0→col 0), O2→D2 (row 1→col 1)
                return (np.array([0, 1]), np.array([0, 1]))
            
            mock_hungarian.side_effect = track_hungarian_call
            
            # ACT - Trigger periodic assignment
            assignment_service.perform_periodic_assignment()
        
        # ASSERT
        # Verify Hungarian algorithm was called
        assert len(hungarian_calls) == 1, "Hungarian algorithm should be called once"
        
        call_data = hungarian_calls[0]
        
        # Verify maximize=True was passed (score maximization, not cost minimization)
        assert call_data["maximize"] is True, "Should maximize scores, not minimize costs"
        
        # Verify correct matrix was passed
        passed_matrix = call_data["matrix"]
        assert passed_matrix.shape == (2, 2), "Matrix should be 2×2"
        np.testing.assert_array_equal(passed_matrix, score_matrix)
        
        # Verify assignments were created based on optimal solution
        delivery_units = delivery_unit_repo.find_all()
        assert len(delivery_units) == 2, "Should create 2 delivery units"
        
        # Check that optimal assignments were made
        assignments = {du.delivery_entity.order_id: du.driver.driver_id for du in delivery_units}
        assert assignments["O1"] == "D1", "Order O1 should be assigned to Driver D1"
        assert assignments["O2"] == "D2", "Order O2 should be assigned to Driver D2"
    
    # ===== Test 4: Score Consistency =====
    
    def test_periodic_assignment_score_consistency_with_immediate(self, test_environment, test_config, mock_priority_scorer):
        """
        Test that periodic assignment calculates the same scores as immediate assignment.
        
        This is crucial for fairness - an assignment shouldn't have different scores
        depending on whether it goes through immediate or periodic assignment.
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
        
        # Create a simple scenario
        order = Order("O1", [3, 3], [5, 5], env.now - 10)  # 10 minutes old
        order.entity_type = EntityType.ORDER
        order.state = OrderState.CREATED
        order_repo.add(order)
        
        driver = Driver("D1", [1, 1], env.now, 120)
        driver.entity_type = EntityType.DRIVER
        driver.state = DriverState.AVAILABLE
        driver_repo.add(driver)
        
        # Set up consistent score return
        expected_score = 82.0
        expected_components = {
            "distance_score": 0.8,
            "throughput_score": 0.0,
            "fairness_score": 0.85,
            "combined_score_0_1": 0.82,
            "total_distance": 6.5,
            "num_orders": 1,
            "wait_time_minutes": 10.0
        }
        mock_priority_scorer.calculate_priority_score.return_value = (expected_score, expected_components)
        
        # First, calculate score using immediate assignment approach
        immediate_score, immediate_components = mock_priority_scorer.calculate_priority_score(driver, order)
        
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
        periodic_score = unit.assignment_scores['priority_score_0_100']
        
        # The scores should be identical
        assert abs(periodic_score - immediate_score) < 0.01, \
            f"Periodic score ({periodic_score}) should match immediate score ({immediate_score})"
        
        # Verify all score components match
        assert unit.assignment_scores['distance_score'] == immediate_components['distance_score']
        assert unit.assignment_scores['fairness_score'] == immediate_components['fairness_score']
        assert unit.assignment_scores['total_distance'] == immediate_components['total_distance']
    
    # ===== Test 5: Empty Scenarios =====
    
    def test_periodic_handles_empty_scenarios_gracefully(self, test_environment, test_config, mock_priority_scorer):
        """
        Test that periodic assignment handles edge cases with no entities or drivers.
        
        This ensures the system is robust when there's nothing to optimize.
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
        
        # Start with completely empty repositories
        
        # ACT - Attempt periodic assignment with no entities or drivers
        assignment_service.perform_periodic_assignment()
        
        # ASSERT - Should handle gracefully without errors
        assert len(delivery_unit_repo.find_all()) == 0, "No assignments should be created"
        mock_priority_scorer.calculate_priority_score.assert_not_called()
        
        # Test with entities but no drivers
        order = Order("O1", [1, 1], [2, 2], env.now)
        order.entity_type = EntityType.ORDER
        order.state = OrderState.CREATED
        order_repo.add(order)
        
        # ACT - Periodic assignment with entities but no drivers
        assignment_service.perform_periodic_assignment()
        
        # ASSERT - Should handle gracefully
        assert len(delivery_unit_repo.find_all()) == 0, "No assignments should be created without drivers"
        mock_priority_scorer.calculate_priority_score.assert_not_called()
        
        # Test with drivers but no entities
        order_repo._orders.clear()  # Clear orders
        driver = Driver("D1", [0, 0], env.now, 120)
        driver.entity_type = EntityType.DRIVER
        driver.state = DriverState.AVAILABLE
        driver_repo.add(driver)
        
        # ACT - Periodic assignment with drivers but no entities
        assignment_service.perform_periodic_assignment()
        
        # ASSERT - Should handle gracefully
        assert len(delivery_unit_repo.find_all()) == 0, "No assignments should be created without entities"
        mock_priority_scorer.calculate_priority_score.assert_not_called()


