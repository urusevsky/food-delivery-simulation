# tests/unit/utils/test_priority_scoring.py
"""
Unit tests for Priority Scoring System

This module tests the multi-criteria scoring system that evaluates
driver-entity assignment opportunities based on distance efficiency,
throughput optimization, and fairness considerations.
"""

import pytest
import simpy
from unittest.mock import Mock, patch

from delivery_sim.utils.priority_scoring import PriorityScorer, create_priority_scorer
from delivery_sim.entities.order import Order
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.pair import Pair
from delivery_sim.utils.entity_type_utils import EntityType
from delivery_sim.entities.states import OrderState, DriverState


class TestPriorityScorer:
    """Test suite for the PriorityScorer class."""
    
    @pytest.fixture
    def mock_scoring_config(self):
        """Create a mock scoring configuration for testing."""
        config = Mock()
        config.max_distance_ratio_multiplier = 2.0
        config.max_acceptable_wait = 30.0
        config.max_orders_per_trip = 2
        config.weight_distance = 1/3
        config.weight_throughput = 1/3
        config.weight_fairness = 1/3
        return config
    
    @pytest.fixture
    def test_env(self):
        """Create a SimPy environment for testing."""
        return simpy.Environment()
    
    @pytest.fixture
    def priority_scorer(self, mock_scoring_config, test_env):
        """Create a PriorityScorer instance for testing."""
        typical_distance = 5.0  # 5km typical distance
        return PriorityScorer(mock_scoring_config, typical_distance, test_env)
    
    @pytest.fixture
    def sample_driver(self):
        """Create a sample driver for testing."""
        driver = Mock(spec=Driver)
        driver.driver_id = "D1"
        driver.entity_type = EntityType.DRIVER
        driver.location = [0, 0]
        driver.state = DriverState.AVAILABLE
        return driver
    
    @pytest.fixture
    def sample_order(self):
        """Create a sample order for testing."""
        order = Mock(spec=Order)
        order.order_id = "O1"
        order.entity_type = EntityType.ORDER
        order.restaurant_location = [3, 4]  # Restaurant location  
        order.customer_location = [6, 8]    # Customer location
        order.arrival_time = 0
        order.state = OrderState.CREATED
        return order
    
    @pytest.fixture
    def sample_pair(self):
        """Create a sample pair for testing."""
        order1 = Mock(spec=Order)
        order1.order_id = "O1"
        order1.entity_type = EntityType.ORDER
        order1.restaurant_location = [1, 1]
        order1.customer_location = [2, 2]
        order1.arrival_time = 5
        
        order2 = Mock(spec=Order)
        order2.order_id = "O2" 
        order2.entity_type = EntityType.ORDER
        order2.restaurant_location = [3, 3]
        order2.customer_location = [4, 4]
        order2.arrival_time = 10
        
        pair = Mock(spec=Pair)
        pair.pair_id = "P-O1_O2"
        pair.entity_type = EntityType.PAIR
        pair.order1 = order1
        pair.order2 = order2
        # Add required attributes for pairs
        pair.optimal_sequence = [[1, 1], [2, 2], [3, 3], [4, 4]]  # Optimal route sequence
        pair.optimal_cost = 4.0  # Pre-calculated optimal distance
        return pair

    # Test Group 1: Initialization tests
    def test_priority_scorer_initialization(self, mock_scoring_config, test_env):
        """Test that PriorityScorer initializes correctly."""
        # ARRANGE
        typical_distance = 5.0
        
        # ACT
        scorer = PriorityScorer(mock_scoring_config, typical_distance, test_env)
        
        # ASSERT
        assert scorer.config is mock_scoring_config
        assert scorer.typical_distance == typical_distance
        assert scorer.env is test_env
        assert scorer.logger is not None

    # Test Group 2: Distance score calculation tests
    @patch('delivery_sim.utils.priority_scoring.calculate_distance')
    def test_calculate_distance_score_normal_distance(self, mock_calculate_distance, priority_scorer, sample_driver, sample_order):
        """Test distance score calculation for normal distances."""
        # ARRANGE
        # Mock distance calculations: driver -> restaurant (5km), restaurant -> customer (5km)
        mock_calculate_distance.side_effect = [5.0, 5.0]
        total_distance = 10.0
        distance_ratio = total_distance / priority_scorer.typical_distance  # 10.0 / 5.0 = 2.0
        # Since ratio = 2.0 and max_ratio = 2.0, score = 0.0
        
        # ACT
        score = priority_scorer._calculate_distance_score(sample_driver, sample_order)
        
        # ASSERT
        assert score == pytest.approx(0.0)
        assert mock_calculate_distance.call_count == 2

    @patch('delivery_sim.utils.priority_scoring.calculate_distance')
    def test_calculate_distance_score_short_distance(self, mock_calculate_distance, priority_scorer, sample_driver, sample_order):
        """Test distance score calculation for short distances (high score)."""
        # ARRANGE
        # Mock distance calculations: driver -> restaurant (1km), restaurant -> customer (1km)
        mock_calculate_distance.side_effect = [1.0, 1.0]
        total_distance = 2.0
        distance_ratio = total_distance / priority_scorer.typical_distance  # 2.0 / 5.0 = 0.4
        # Since ratio <= 1.0, score = 1.0
        
        # ACT
        score = priority_scorer._calculate_distance_score(sample_driver, sample_order)
        
        # ASSERT
        assert score == pytest.approx(1.0)

    @patch('delivery_sim.utils.priority_scoring.calculate_distance')
    def test_calculate_distance_score_very_long_distance(self, mock_calculate_distance, priority_scorer, sample_driver, sample_order):
        """Test distance score calculation for very long distances (score = 0)."""
        # ARRANGE
        # Mock distance calculations: driver -> restaurant (10km), restaurant -> customer (5km)
        mock_calculate_distance.side_effect = [10.0, 5.0]
        total_distance = 15.0
        distance_ratio = total_distance / priority_scorer.typical_distance  # 15.0 / 5.0 = 3.0
        # Since ratio >= max_ratio (2.0), score = 0.0
        
        # ACT
        score = priority_scorer._calculate_distance_score(sample_driver, sample_order)
        
        # ASSERT
        assert score == 0.0

    # Test Group 3: Throughput score calculation tests  
    def test_calculate_throughput_score_single_order(self, priority_scorer, sample_order):
        """Test throughput score calculation for a single order."""
        # ARRANGE
        # Single order should give score of (1-1)/(2-1) = 0.0
        
        # ACT
        score = priority_scorer._calculate_throughput_score(sample_order)
        
        # ASSERT
        assert score == 0.0

    def test_calculate_throughput_score_pair(self, priority_scorer, sample_pair):
        """Test throughput score calculation for a pair."""
        # ARRANGE
        # Pair has 2 orders, should give score of (2-1)/(2-1) = 1.0
        
        # ACT
        score = priority_scorer._calculate_throughput_score(sample_pair)
        
        # ASSERT
        assert score == 1.0

    # Test Group 4: Fairness score calculation tests
    def test_calculate_fairness_score_no_wait(self, priority_scorer, sample_order, test_env):
        """Test fairness score calculation when there's no wait time."""
        # ARRANGE
        sample_order.arrival_time = test_env.now  # No wait time
        
        # ACT
        score = priority_scorer._calculate_fairness_score(sample_order)
        
        # ASSERT
        assert score == 1.0  # No wait time = perfect score

    def test_calculate_fairness_score_moderate_wait(self, priority_scorer, sample_order, test_env):
        """Test fairness score calculation with moderate wait time."""
        # ARRANGE
        test_env.run(until=15)  # Advance time to 15 minutes
        sample_order.arrival_time = 0  # Order arrived at time 0
        # Wait time = 15 minutes, max acceptable = 30 minutes
        expected_score = 1.0 - (15.0 / 30.0)  # 1.0 - 0.5 = 0.5
        
        # ACT
        score = priority_scorer._calculate_fairness_score(sample_order)
        
        # ASSERT
        assert score == pytest.approx(0.5)

    def test_calculate_fairness_score_long_wait(self, priority_scorer, sample_order, test_env):
        """Test fairness score calculation with long wait time (capped at 0.0)."""
        # ARRANGE
        test_env.run(until=45)  # Advance time to 45 minutes
        sample_order.arrival_time = 0  # Order arrived at time 0
        # Wait time = 45 minutes, max acceptable = 30 minutes
        expected_score = max(0.0, 1.0 - (45.0 / 30.0))  # max(0.0, -0.5) = 0.0
        
        # ACT
        score = priority_scorer._calculate_fairness_score(sample_order)
        
        # ASSERT
        assert score == 0.0

    def test_calculate_fairness_score_pair_uses_earliest_arrival(self, priority_scorer, sample_pair, test_env):
        """Test that pair fairness score uses the earliest order arrival time."""
        # ARRANGE
        test_env.run(until=20)  # Current time = 20
        sample_pair.order1.arrival_time = 5   # Earlier order
        sample_pair.order2.arrival_time = 10  # Later order
        # Should use earliest (5), so wait time = 20 - 5 = 15 minutes
        expected_score = 1.0 - (15.0 / 30.0)  # 1.0 - 0.5 = 0.5
        
        # ACT
        score = priority_scorer._calculate_fairness_score(sample_pair)
        
        # ASSERT
        assert score == pytest.approx(0.5)

    # Test Group 5: Helper method tests
    def test_get_order_count_single_order(self, priority_scorer, sample_order):
        """Test _get_order_count returns 1 for single orders."""
        # ACT
        num_orders = priority_scorer._get_order_count(sample_order)
        
        # ASSERT
        assert num_orders == 1

    def test_get_order_count_pair(self, priority_scorer, sample_pair):
        """Test _get_order_count returns 2 for pairs."""
        # ACT
        num_orders = priority_scorer._get_order_count(sample_pair)
        
        # ASSERT
        assert num_orders == 2

    def test_calculate_wait_time_single_order(self, priority_scorer, sample_order, test_env):
        """Test wait time calculation for single orders."""
        # ARRANGE
        test_env.run(until=10)
        sample_order.arrival_time = 3
        
        # ACT
        wait_time = priority_scorer._calculate_wait_time(sample_order)
        
        # ASSERT
        assert wait_time == 7.0  # 10 - 3

    def test_calculate_wait_time_pair(self, priority_scorer, sample_pair, test_env):
        """Test wait time calculation for pairs uses earliest arrival."""
        # ARRANGE
        test_env.run(until=20)
        sample_pair.order1.arrival_time = 5   # Earlier
        sample_pair.order2.arrival_time = 10  # Later
        
        # ACT
        wait_time = priority_scorer._calculate_wait_time(sample_pair)
        
        # ASSERT
        assert wait_time == 15.0  # 20 - 5 (uses earliest)

    # Test Group 6: Integration tests for full score calculation
    @patch('delivery_sim.utils.priority_scoring.calculate_distance')
    def test_calculate_priority_score_integration(self, mock_calculate_distance, priority_scorer, sample_driver, sample_order, test_env):
        """Test full priority score calculation integration."""
        # ARRANGE
        test_env.run(until=15)  # Set current time
        sample_order.arrival_time = 0  # Order has been waiting 15 minutes
        
        # Mock distance calculations (total 5km, same as typical)
        # _calculate_total_distance gets called twice, each time makes 2 distance calls
        mock_calculate_distance.side_effect = [2.5, 2.5, 2.5, 2.5]  # 4 calls total
        
        # Expected component scores:
        # - Distance: ratio = 5/5 = 1.0, since ratio <= 1.0, score = 1.0
        # - Throughput: (1-1)/(2-1) = 0.0
        # - Fairness: 1.0 - (15/30) = 0.5
        # Combined (0-1): (1/3)*1.0 + (1/3)*0.0 + (1/3)*0.5 = 0.5
        # Priority (0-100): 0.5 * 100 = 50.0
        
        # ACT
        score, components = priority_scorer.calculate_priority_score(sample_driver, sample_order)
        
        # ASSERT
        assert score == pytest.approx(50.0)
        assert components["distance_score"] == pytest.approx(1.0)
        assert components["throughput_score"] == pytest.approx(0.0)
        assert components["fairness_score"] == pytest.approx(0.5)
        assert components["combined_score_0_1"] == pytest.approx(0.5)
        assert components["total_distance"] == pytest.approx(5.0)
        assert components["num_orders"] == 1
        assert components["wait_time_minutes"] == pytest.approx(15.0)

    @patch('delivery_sim.utils.priority_scoring.calculate_distance')
    def test_calculate_priority_score_with_pair(self, mock_calculate_distance, priority_scorer, sample_driver, sample_pair, test_env):
        """Test priority score calculation with a pair entity."""
        # ARRANGE
        test_env.run(until=20)
        sample_pair.order1.arrival_time = 5
        sample_pair.order2.arrival_time = 10
        
        # Mock distance calculation for pair: driver to first location only
        # _calculate_total_distance gets called twice, each time makes 1 distance call for pairs
        mock_calculate_distance.side_effect = [1.0, 1.0]  # 2 calls total
        # Total distance = 1.0 + 4.0 (optimal_cost) = 5.0km
        
        # Expected component scores:
        # - Distance: ratio = 5/5 = 1.0, since ratio <= 1.0, score = 1.0
        # - Throughput: (2-1)/(2-1) = 1.0
        # - Fairness: 1.0 - ((20-5)/30) = 1.0 - 0.5 = 0.5 (uses earliest arrival)
        # Combined (0-1): (1/3)*1.0 + (1/3)*1.0 + (1/3)*0.5 = 5/6 ≈ 0.833
        # Priority (0-100): (5/6) * 100 ≈ 83.33
        
        # ACT
        score, components = priority_scorer.calculate_priority_score(sample_driver, sample_pair)
        
        # ASSERT
        assert score == pytest.approx(83.33, abs=0.1)
        assert components["distance_score"] == pytest.approx(1.0)
        assert components["throughput_score"] == pytest.approx(1.0)
        assert components["fairness_score"] == pytest.approx(0.5)
        assert components["num_orders"] == 2

    # Test Group 7: Edge cases and error handling
    def test_calculate_priority_score_zero_typical_distance(self, mock_scoring_config, test_env, sample_driver, sample_order):
        """Test that zero typical distance is handled gracefully."""
        # ARRANGE
        scorer = PriorityScorer(mock_scoring_config, 0.001, test_env)  # Very small typical distance
        
        # ACT & ASSERT - Should not raise ZeroDivisionError
        with patch('delivery_sim.utils.priority_scoring.calculate_distance', return_value=1.0):
            score, components = scorer.calculate_priority_score(sample_driver, sample_order)
            assert isinstance(score, float)
            assert score >= 0


class TestCreatePriorityScorer:
    """Test suite for the create_priority_scorer factory function."""
    
    def test_create_priority_scorer_factory(self):
        """Test that the factory function creates a properly configured scorer."""
        # ARRANGE
        infrastructure_characteristics = {'typical_distance': 7.5}
        mock_scoring_config = Mock()
        test_env = simpy.Environment()
        
        # ACT
        scorer = create_priority_scorer(infrastructure_characteristics, mock_scoring_config, test_env)
        
        # ASSERT
        assert isinstance(scorer, PriorityScorer)
        assert scorer.typical_distance == 7.5
        assert scorer.config is mock_scoring_config
        assert scorer.env is test_env