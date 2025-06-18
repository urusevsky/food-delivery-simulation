# tests/unit/utils/test_priority_scoring.py
"""
Updated tests for PriorityScorer with corrected implementation.

Major fixes:
1. Distance Score: Now uses linear formula max(0, 1 - ratio/max_multiplier) instead of plateau
2. Throughput Score: Single orders now score 0.5 (num_orders/max_orders) instead of 0.0  
3. Fairness Score: Now uses min(1.0, wait_time/max_wait) for urgency instead of backwards formula
4. Method Signatures: _calculate_distance_score now takes total_distance directly
"""

import pytest
import simpy
from unittest.mock import Mock, patch
from delivery_sim.utils.priority_scoring import PriorityScorer, create_priority_scorer
from delivery_sim.entities.order import Order
from delivery_sim.entities.pair import Pair
from delivery_sim.entities.driver import Driver


class TestPriorityScorer:
    """Test suite for the PriorityScorer class with corrected implementation."""

    @pytest.fixture
    def test_env(self):
        """Create a test SimPy environment."""
        return simpy.Environment()

    @pytest.fixture  
    def mock_scoring_config(self):
        """Create a mock scoring configuration."""
        config = Mock()
        config.max_distance_ratio_multiplier = 2.0  # Distances >2x typical = score 0
        config.max_acceptable_wait = 30.0  # 30 min = maximum urgency  
        config.max_orders_per_trip = 2  # System constraint
        config.weight_distance = 1/3
        config.weight_throughput = 1/3  
        config.weight_fairness = 1/3
        return config

    @pytest.fixture
    def priority_scorer(self, mock_scoring_config, test_env):
        """Create a PriorityScorer with test configuration."""
        typical_distance = 5.0  # 5km typical for this test environment
        return PriorityScorer(mock_scoring_config, typical_distance, test_env)

    @pytest.fixture
    def sample_driver(self):
        """Create a sample driver for testing."""
        return Driver("D1", [0, 0], 0, 120)

    @pytest.fixture
    def sample_order(self):
        """Create a sample order for testing."""
        return Order("O1", [1, 1], [2, 2], 0)

    @pytest.fixture
    def sample_pair(self):
        """Create a sample pair for testing."""
        order1 = Order("O1", [1, 1], [2, 2], 0)
        order2 = Order("O2", [3, 3], [4, 4], 5) 
        pair = Pair(order1, order2, 10)
        # Mock the optimal sequence and cost for pair testing
        pair.optimal_sequence = [[1, 1], [2, 2], [3, 3], [4, 4]]
        pair.optimal_cost = 4.0  # Pre-calculated optimal travel cost
        return pair

    # Test Group 1: Distance score calculation (CORRECTED)
    def test_calculate_distance_score_perfect_efficiency(self, priority_scorer):
        """Test distance score for zero distance (perfect efficiency)."""
        # ARRANGE
        total_distance = 0.0
        
        # ACT
        score = priority_scorer._calculate_distance_score(total_distance)
        
        # ASSERT  
        assert score == 1.0  # Perfect efficiency

    def test_calculate_distance_score_typical_distance(self, priority_scorer):
        """Test distance score for typical distance."""
        # ARRANGE
        total_distance = 5.0  # Same as typical_distance
        # ratio = 5.0/5.0 = 1.0, score = max(0, 1 - 1.0/2.0) = 0.5
        
        # ACT
        score = priority_scorer._calculate_distance_score(total_distance)
        
        # ASSERT
        assert score == pytest.approx(0.5)  # 50% efficiency at typical distance

    def test_calculate_distance_score_excellent_efficiency(self, priority_scorer):
        """Test distance score for short distances (excellent efficiency)."""
        # ARRANGE
        total_distance = 2.5  # Half of typical distance
        # ratio = 2.5/5.0 = 0.5, score = max(0, 1 - 0.5/2.0) = 0.75
        
        # ACT
        score = priority_scorer._calculate_distance_score(total_distance)
        
        # ASSERT
        assert score == pytest.approx(0.75)  # 75% efficiency

    def test_calculate_distance_score_unacceptable_distance(self, priority_scorer):
        """Test distance score for unacceptably long distances."""
        # ARRANGE  
        total_distance = 10.0  # 2x typical distance (max ratio)
        # ratio = 10.0/5.0 = 2.0, score = max(0, 1 - 2.0/2.0) = 0.0
        
        # ACT
        score = priority_scorer._calculate_distance_score(total_distance)
        
        # ASSERT
        assert score == 0.0  # Unacceptable efficiency

    def test_calculate_distance_score_beyond_max(self, priority_scorer):
        """Test distance score for distances beyond maximum ratio."""
        # ARRANGE
        total_distance = 15.0  # 3x typical distance (beyond max)
        # ratio = 15.0/5.0 = 3.0, score = max(0, 1 - 3.0/2.0) = max(0, -0.5) = 0.0
        
        # ACT
        score = priority_scorer._calculate_distance_score(total_distance)
        
        # ASSERT
        assert score == 0.0  # Still capped at 0.0

    # Test Group 2: Throughput score calculation (CORRECTED)
    def test_calculate_throughput_score_single_order(self, priority_scorer, sample_order):
        """Test throughput score calculation for a single order."""
        # ARRANGE
        # Single order: num_orders=1, max_orders=2, score = 1/2 = 0.5
        
        # ACT
        score = priority_scorer._calculate_throughput_score(sample_order)
        
        # ASSERT
        assert score == 0.5  # 50% capacity utilization

    def test_calculate_throughput_score_pair(self, priority_scorer, sample_pair):
        """Test throughput score calculation for a pair."""
        # ARRANGE
        # Pair: num_orders=2, max_orders=2, score = 2/2 = 1.0
        
        # ACT
        score = priority_scorer._calculate_throughput_score(sample_pair)
        
        # ASSERT
        assert score == 1.0  # 100% capacity utilization

    # Test Group 3: Fairness score calculation (CORRECTED)
    def test_calculate_fairness_score_no_wait(self, priority_scorer, sample_order, test_env):
        """Test fairness score calculation when there's no wait time."""
        # ARRANGE
        sample_order.arrival_time = test_env.now  # No wait time
        
        # ACT
        score = priority_scorer._calculate_fairness_score(sample_order)
        
        # ASSERT
        assert score == 0.0  # No wait = no urgency

    def test_calculate_fairness_score_moderate_wait(self, priority_scorer, sample_order, test_env):
        """Test fairness score calculation with moderate wait time."""
        # ARRANGE
        test_env.run(until=15)  # Advance time to 15 minutes
        sample_order.arrival_time = 0  # Order arrived at time 0
        # Wait time = 15 minutes, max_wait = 30 minutes
        # Score = min(1.0, 15.0/30.0) = 0.5
        
        # ACT
        score = priority_scorer._calculate_fairness_score(sample_order)
        
        # ASSERT
        assert score == pytest.approx(0.5)  # Moderate urgency

    def test_calculate_fairness_score_maximum_urgency(self, priority_scorer, sample_order, test_env):
        """Test fairness score calculation with maximum urgency wait time."""
        # ARRANGE
        test_env.run(until=30)  # Advance time to 30 minutes  
        sample_order.arrival_time = 0  # Order arrived at time 0
        # Wait time = 30 minutes, max_wait = 30 minutes
        # Score = min(1.0, 30.0/30.0) = 1.0
        
        # ACT
        score = priority_scorer._calculate_fairness_score(sample_order)
        
        # ASSERT
        assert score == 1.0  # Maximum urgency

    def test_calculate_fairness_score_beyond_max_wait(self, priority_scorer, sample_order, test_env):
        """Test fairness score calculation beyond maximum wait time."""
        # ARRANGE
        test_env.run(until=45)  # Advance time to 45 minutes
        sample_order.arrival_time = 0  # Order arrived at time 0  
        # Wait time = 45 minutes, max_wait = 30 minutes
        # Score = min(1.0, 45.0/30.0) = min(1.0, 1.5) = 1.0
        
        # ACT
        score = priority_scorer._calculate_fairness_score(sample_order)
        
        # ASSERT
        assert score == 1.0  # Capped at maximum urgency

    def test_calculate_fairness_score_pair_uses_earliest_arrival(self, priority_scorer, sample_pair, test_env):
        """Test that pair fairness score uses the earliest order arrival time."""
        # ARRANGE
        test_env.run(until=20)  # Current time = 20
        sample_pair.order1.arrival_time = 5   # Earlier order
        sample_pair.order2.arrival_time = 10  # Later order
        # Should use earliest (5), so wait time = 20 - 5 = 15 minutes
        # Score = min(1.0, 15.0/30.0) = 0.5
        
        # ACT
        score = priority_scorer._calculate_fairness_score(sample_pair)
        
        # ASSERT
        assert score == pytest.approx(0.5)  # Based on earliest arrival

    # Test Group 4: Helper method tests
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

    # Test Group 5: Integration tests for full score calculation (CORRECTED)
    @patch('delivery_sim.utils.priority_scoring.calculate_distance')
    def test_calculate_priority_score_integration_single_order(self, mock_calculate_distance, priority_scorer, sample_driver, sample_order, test_env):
        """Test full priority score calculation integration for single order."""
        # ARRANGE
        test_env.run(until=15)  # Set current time
        sample_order.arrival_time = 0  # Order has been waiting 15 minutes
        
        # Mock distance calculations (total 5km, same as typical)
        mock_calculate_distance.side_effect = [2.5, 2.5]  # driver->restaurant, restaurant->customer
        
        # Expected component scores:
        # - Distance: ratio = 5/5 = 1.0, score = max(0, 1 - 1.0/2.0) = 0.5
        # - Throughput: 1/2 = 0.5  
        # - Fairness: min(1.0, 15/30) = 0.5
        # Combined (0-1): (1/3)*0.5 + (1/3)*0.5 + (1/3)*0.5 = 0.5
        # Priority (0-100): 0.5 * 100 = 50.0
        
        # ACT
        score, components = priority_scorer.calculate_priority_score(sample_driver, sample_order)
        
        # ASSERT
        assert score == pytest.approx(50.0)
        assert components["distance_score"] == pytest.approx(0.5)
        assert components["throughput_score"] == pytest.approx(0.5)
        assert components["fairness_score"] == pytest.approx(0.5)
        assert components["combined_score_0_1"] == pytest.approx(0.5)
        assert components["total_distance"] == pytest.approx(5.0)
        assert components["num_orders"] == 1
        assert components["wait_time_minutes"] == pytest.approx(15.0)

    @patch('delivery_sim.utils.priority_scoring.calculate_distance')
    def test_calculate_priority_score_integration_pair(self, mock_calculate_distance, priority_scorer, sample_driver, sample_pair, test_env):
        """Test full priority score calculation integration for pair."""
        # ARRANGE
        test_env.run(until=20)
        sample_pair.order1.arrival_time = 5
        sample_pair.order2.arrival_time = 10
        
        # Mock distance calculation for pair: driver to first location only
        mock_calculate_distance.return_value = 1.0
        # Total distance = 1.0 + 4.0 (optimal_cost) = 5.0km
        
        # Expected component scores:
        # - Distance: ratio = 5/5 = 1.0, score = max(0, 1 - 1.0/2.0) = 0.5
        # - Throughput: 2/2 = 1.0
        # - Fairness: min(1.0, (20-5)/30) = min(1.0, 0.5) = 0.5 (uses earliest arrival)
        # Combined (0-1): (1/3)*0.5 + (1/3)*1.0 + (1/3)*0.5 = 2/3 ≈ 0.667
        # Priority (0-100): (2/3) * 100 ≈ 66.67
        
        # ACT
        score, components = priority_scorer.calculate_priority_score(sample_driver, sample_pair)
        
        # ASSERT
        assert score == pytest.approx(66.67, abs=0.1)
        assert components["distance_score"] == pytest.approx(0.5)
        assert components["throughput_score"] == pytest.approx(1.0)
        assert components["fairness_score"] == pytest.approx(0.5)
        assert components["num_orders"] == 2

    # Test Group 6: Edge cases and error handling
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