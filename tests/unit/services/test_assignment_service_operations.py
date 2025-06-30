# tests/unit/services/test_assignment_service_operations.py
"""
Updated tests for AssignmentService with priority scoring system.

Major changes from adjusted cost system:
- calculate_adjusted_cost() → priority_scorer.calculate_priority_score()
- _generate_cost_matrix() → _generate_score_matrix()
- _find_best_match() logic reversed: minimize cost → maximize score
- Constructor now requires priority_scorer dependency
- Mock priority scorer instead of cost calculation methods
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from delivery_sim.services.assignment_service import AssignmentService
from delivery_sim.entities.order import Order
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.pair import Pair
from delivery_sim.entities.states import OrderState, DriverState, PairState
from delivery_sim.utils.entity_type_utils import EntityType


class TestAssignmentServiceOperations:
    """Test suite for AssignmentService operations with priority scoring system."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock SimPy environment."""
        env = Mock()
        env.now = 100  # Set a fixed current time for testing
        return env

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration with needed parameters."""
        config = Mock()
        config.immediate_assignment_threshold = 75.0  # Priority score threshold (0-100)
        config.periodic_interval = 3.0  # Minutes between periodic assignments
        config.pairing_enabled = True
        return config

    @pytest.fixture
    def mock_repositories(self):
        """Create mock repositories."""
        return {
            'order': Mock(),
            'driver': Mock(),
            'pair': Mock(),
            'delivery_unit': Mock()
        }

    @pytest.fixture
    def mock_priority_scorer(self):
        """Create a mock priority scorer."""
        scorer = Mock()
        # Set up default return values for scoring
        scorer.calculate_priority_score.return_value = (75.0, {
            "distance_score": 0.8,
            "throughput_score": 0.5,
            "fairness_score": 0.9,
            "combined_score_0_1": 0.75,
            "total_distance": 8.5,
            "num_orders": 1,
            "assignment_delay_minutes": 5.0
        })
        return scorer

    @pytest.fixture
    def service(self, mock_env, mock_repositories, mock_config, mock_priority_scorer):
        """Create an AssignmentService instance with mocked dependencies."""
        event_dispatcher = Mock()
        service = AssignmentService(
            env=mock_env,
            event_dispatcher=event_dispatcher,
            order_repository=mock_repositories['order'],
            driver_repository=mock_repositories['driver'],
            pair_repository=mock_repositories['pair'],
            delivery_unit_repository=mock_repositories['delivery_unit'],
            priority_scorer=mock_priority_scorer,
            config=mock_config
        )
        return service

    @pytest.fixture
    def sample_driver(self):
        """Create a sample driver for testing."""
        driver = Mock(spec=Driver)
        driver.driver_id = "D1"
        driver.location = [0, 0]
        driver.entity_type = EntityType.DRIVER
        driver.state = DriverState.AVAILABLE
        return driver

    @pytest.fixture
    def sample_order(self):
        """Create a sample order for testing."""
        order = Mock(spec=Order)
        order.order_id = "O1"
        order.restaurant_location = [3, 4]
        order.customer_location = [6, 8]
        order.arrival_time = 80  # Arrived 20 minutes ago
        order.entity_type = EntityType.ORDER
        order.state = OrderState.CREATED
        return order

    @pytest.fixture
    def sample_pair(self):
        """Create a sample pair for testing."""
        order1 = Mock(spec=Order)
        order1.order_id = "O1"
        order1.arrival_time = 75
        
        order2 = Mock(spec=Order)
        order2.order_id = "O2"
        order2.arrival_time = 80
        
        pair = Mock(spec=Pair)
        pair.pair_id = "P-O1_O2"
        pair.order1 = order1
        pair.order2 = order2
        pair.entity_type = EntityType.PAIR
        pair.state = PairState.CREATED
        
        # Set optimal sequence and cost
        pair.optimal_sequence = [[3, 4], [6, 8], [9, 12]]
        pair.optimal_cost = 10.0
        
        return pair

    # Test Group 1: _find_best_match method (updated for score maximization)
    def test_find_best_match_from_drivers(self, service, sample_order):
        """Test finding the best driver for an order using priority scores."""
        # ARRANGE
        # Create 3 potential drivers with different locations
        driver1 = Mock(spec=Driver)
        driver1.driver_id = "D1"
        driver1.location = [0, 0]
        driver1.entity_type = EntityType.DRIVER
        
        driver2 = Mock(spec=Driver)
        driver2.driver_id = "D2"
        driver2.location = [2, 2]
        driver2.entity_type = EntityType.DRIVER
        
        driver3 = Mock(spec=Driver)
        driver3.driver_id = "D3"
        driver3.location = [3, 3]
        driver3.entity_type = EntityType.DRIVER
        
        candidates = [driver1, driver2, driver3]
        
        # Mock the priority scorer to return known values
        # Driver1: 60, Driver2: 75, Driver3: 85 (higher scores are better)
        service.priority_scorer.calculate_priority_score.side_effect = [
            (60.0, {"distance_score": 0.6, "total_distance": 12.0}),
            (75.0, {"distance_score": 0.75, "total_distance": 10.0}),
            (85.0, {"distance_score": 0.85, "total_distance": 8.0})
        ]
        
        # ACT
        best_match, best_score, best_components = service._find_best_match(sample_order, candidates)
        
        # ASSERT
        assert best_match is driver3
        assert best_score == pytest.approx(85.0)
        assert best_components["distance_score"] == pytest.approx(0.85)

    def test_find_best_match_from_delivery_entities(self, service, sample_driver):
        """Test finding the best delivery entity for a driver using priority scores."""
        # ARRANGE
        # Create 2 orders and 1 pair with different characteristics
        order1 = Mock(spec=Order)
        order1.order_id = "O1"
        order1.restaurant_location = [5, 5]
        order1.customer_location = [7, 8]
        order1.entity_type = EntityType.ORDER
        
        order2 = Mock(spec=Order)
        order2.order_id = "O2"
        order2.restaurant_location = [2, 3]
        order2.customer_location = [4, 6]
        order2.entity_type = EntityType.ORDER
        
        pair = Mock(spec=Pair)
        pair.pair_id = "P-O3_O4"
        pair.optimal_sequence = [[1, 1], [3, 3], [5, 5]]
        pair.entity_type = EntityType.PAIR
        
        candidates = [order1, order2, pair]
        
        # Mock the priority scorer to return known values
        # Order1: 70, Order2: 85, Pair: 80 (higher scores are better)
        service.priority_scorer.calculate_priority_score.side_effect = [
            (70.0, {"distance_score": 0.7, "throughput_score": 0.0}),
            (85.0, {"distance_score": 0.85, "throughput_score": 0.0}),
            (80.0, {"distance_score": 0.6, "throughput_score": 1.0})
        ]
        
        # ACT
        best_match, best_score, best_components = service._find_best_match(sample_driver, candidates)
        
        # ASSERT
        assert best_match is order2
        assert best_score == pytest.approx(85.0)
        assert best_components["distance_score"] == pytest.approx(0.85)

    def test_find_best_match_empty_candidates(self, service, sample_driver):
        """Test finding the best match when there are no candidates."""
        # ARRANGE
        candidates = []
        
        # ACT
        best_match, best_score, best_components = service._find_best_match(sample_driver, candidates)
        
        # ASSERT
        assert best_match is None
        assert best_score == -1.0  # Sentinel value indicating no candidates evaluated
        assert best_components is None

    # Test Group 2: _generate_score_matrix method (renamed from _generate_cost_matrix)
    def test_generate_score_matrix(self, service):
        """Test generating score matrix for optimization."""
        # ARRANGE
        # Create 2 waiting entities and 2 drivers
        order1 = Mock(spec=Order)
        order1.order_id = "O1"
        order1.entity_type = EntityType.ORDER
        
        pair1 = Mock(spec=Pair)
        pair1.pair_id = "P-O2_O3"
        pair1.entity_type = EntityType.PAIR
        
        driver1 = Mock(spec=Driver)
        driver1.driver_id = "D1"
        driver1.entity_type = EntityType.DRIVER
        
        driver2 = Mock(spec=Driver)
        driver2.driver_id = "D2"
        driver2.entity_type = EntityType.DRIVER
        
        waiting_entities = [order1, pair1]
        available_drivers = [driver1, driver2]
        
        # Mock the priority scorer to return known values
        service.priority_scorer.calculate_priority_score.side_effect = [
            (65.0, {}),  # order1-driver1
            (70.0, {}),  # order1-driver2
            (80.0, {}),  # pair1-driver1
            (75.0, {})   # pair1-driver2
        ]
        
        # ACT
        score_matrix = service._generate_score_matrix(waiting_entities, available_drivers)
        
        # ASSERT
        assert score_matrix.shape == (2, 2)  # Two waiting entities, two drivers
        assert score_matrix[0, 0] == pytest.approx(65.0)  # order1-driver1
        assert score_matrix[0, 1] == pytest.approx(70.0)  # order1-driver2
        assert score_matrix[1, 0] == pytest.approx(80.0)  # pair1-driver1
        assert score_matrix[1, 1] == pytest.approx(75.0)  # pair1-driver2

    def test_generate_score_matrix_single_entity_multiple_drivers(self, service):
        """Test generating score matrix with one entity and multiple drivers."""
        # ARRANGE
        order = Mock(spec=Order)
        order.order_id = "O1"
        order.entity_type = EntityType.ORDER
        
        driver1 = Mock(spec=Driver)
        driver1.driver_id = "D1"
        driver1.entity_type = EntityType.DRIVER
        
        driver2 = Mock(spec=Driver)
        driver2.driver_id = "D2"
        driver2.entity_type = EntityType.DRIVER
        
        driver3 = Mock(spec=Driver)
        driver3.driver_id = "D3"
        driver3.entity_type = EntityType.DRIVER
        
        waiting_entities = [order]
        available_drivers = [driver1, driver2, driver3]
        
        # Mock the priority scorer to return known values
        service.priority_scorer.calculate_priority_score.side_effect = [
            (65.0, {}),  # order-driver1
            (75.0, {}),  # order-driver2
            (70.0, {})   # order-driver3
        ]
        
        # ACT
        score_matrix = service._generate_score_matrix(waiting_entities, available_drivers)
        
        # ASSERT
        assert score_matrix.shape == (1, 3)  # One waiting entity, three drivers
        assert score_matrix[0, 0] == pytest.approx(65.0)  # order-driver1
        assert score_matrix[0, 1] == pytest.approx(75.0)  # order-driver2
        assert score_matrix[0, 2] == pytest.approx(70.0)  # order-driver3

    def test_generate_score_matrix_multiple_entities_single_driver(self, service):
        """Test generating score matrix with multiple entities and one driver."""
        # ARRANGE
        order1 = Mock(spec=Order)
        order1.order_id = "O1"
        order1.entity_type = EntityType.ORDER
        
        order2 = Mock(spec=Order)
        order2.order_id = "O2"
        order2.entity_type = EntityType.ORDER
        
        pair1 = Mock(spec=Pair)
        pair1.pair_id = "P-O3_O4"
        pair1.entity_type = EntityType.PAIR
        
        driver = Mock(spec=Driver)
        driver.driver_id = "D1"
        driver.entity_type = EntityType.DRIVER
        
        waiting_entities = [order1, order2, pair1]
        available_drivers = [driver]
        
        # Mock the priority scorer to return known values
        service.priority_scorer.calculate_priority_score.side_effect = [
            (65.0, {}),  # order1-driver
            (75.0, {}),  # order2-driver
            (80.0, {})   # pair1-driver
        ]
        
        # ACT
        score_matrix = service._generate_score_matrix(waiting_entities, available_drivers)
        
        # ASSERT
        assert score_matrix.shape == (3, 1)  # Three waiting entities, one driver
        assert score_matrix[0, 0] == pytest.approx(65.0)  # order1-driver
        assert score_matrix[1, 0] == pytest.approx(75.0)  # order2-driver
        assert score_matrix[2, 0] == pytest.approx(80.0)  # pair1-driver

    # Test Group 3: Integration tests for priority scorer usage
    def test_priority_scorer_integration_in_assignment_evaluation(self, service, sample_driver, sample_order):
        """Test that the assignment service correctly integrates with the priority scorer."""
        # ARRANGE
        expected_score = 82.5
        expected_components = {
            "distance_score": 0.8,
            "throughput_score": 0.0,
            "fairness_score": 0.9,
            "combined_score_0_1": 0.825,
            "total_distance": 7.5,
            "num_orders": 1,
            "assignment_delay_minutes": 3.0
        }
        
        service.priority_scorer.calculate_priority_score.return_value = (expected_score, expected_components)
        
        # ACT
        candidates = [sample_driver]
        best_match, best_score, best_components = service._find_best_match(sample_order, candidates)
        
        # ASSERT
        # Verify priority scorer was called correctly
        service.priority_scorer.calculate_priority_score.assert_called_once_with(sample_driver, sample_order)
        
        # Verify results
        assert best_match is sample_driver
        assert best_score == expected_score
        assert best_components == expected_components

    def test_service_handles_multiple_priority_score_calculations(self, service):
        """Test that the service can handle multiple priority score calculations efficiently."""
        # ARRANGE
        entities = [Mock(entity_type=EntityType.ORDER, order_id=f"O{i}") for i in range(3)]
        drivers = [Mock(entity_type=EntityType.DRIVER, driver_id=f"D{i}") for i in range(2)]
        
        # Mock score calculations for all combinations (3 entities × 2 drivers = 6 calls)
        scores = [65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
        service.priority_scorer.calculate_priority_score.side_effect = [
            (score, {"combined_score_0_1": score/100}) for score in scores
        ]
        
        # ACT
        score_matrix = service._generate_score_matrix(entities, drivers)
        
        # ASSERT
        assert service.priority_scorer.calculate_priority_score.call_count == 6
        assert score_matrix.shape == (3, 2)
        # Verify scores are correctly placed in matrix
        assert score_matrix[0, 0] == 65.0  # entity0-driver0
        assert score_matrix[2, 1] == 90.0  # entity2-driver1

    # Test Group 4: Error handling and edge cases
    def test_priority_scorer_error_handling(self, service, sample_driver, sample_order):
        """Test that the service handles priority scorer errors gracefully."""
        # ARRANGE
        service.priority_scorer.calculate_priority_score.side_effect = Exception("Scorer error")
        
        # ACT & ASSERT
        with pytest.raises(Exception, match="Scorer error"):
            service._find_best_match(sample_order, [sample_driver])

    def test_service_with_zero_scores(self, service, sample_driver, sample_order):
        """Test service behavior when all priority scores are zero."""
        # ARRANGE
        service.priority_scorer.calculate_priority_score.return_value = (0.0, {"combined_score_0_1": 0.0})
        
        # ACT
        best_match, best_score, best_components = service._find_best_match(sample_order, [sample_driver])
        
        # ASSERT
        assert best_match is sample_driver  # Still returns the only available option
        assert best_score == 0.0
        assert best_components["combined_score_0_1"] == 0.0