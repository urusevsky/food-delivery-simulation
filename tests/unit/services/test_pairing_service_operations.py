# tests/unit/services/test_pairing_service_operations.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from delivery_sim.services.pairing_service import PairingService
from delivery_sim.entities.order import Order
from delivery_sim.entities.states import OrderState
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.repositories.order_repository import OrderRepository
from delivery_sim.repositories.pair_repository import PairRepository


class TestPairingServiceOperations:
    """Test suite for PairingService operations that can be tested in isolation."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock SimPy environment."""
        env = Mock()
        env.now = 100.0  # Set a fixed current time for testing
        return env

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration with needed parameters."""
        config = Mock()
        config.restaurants_proximity_threshold = 2.0  # km
        config.customers_proximity_threshold = 3.0  # km
        return config

    @pytest.fixture
    def mock_repositories(self):
        """Create mock repositories."""
        order_repo = Mock(spec=OrderRepository)
        pair_repo = Mock(spec=PairRepository)
        return {
            'order_repository': order_repo,
            'pair_repository': pair_repo
        }

    @pytest.fixture
    def service(self, mock_env, mock_repositories, mock_config):
        """Create a PairingService instance with mocked dependencies."""
        event_dispatcher = Mock(spec=EventDispatcher)
        service = PairingService(
            env=mock_env,
            event_dispatcher=event_dispatcher,
            order_repository=mock_repositories['order_repository'],
            pair_repository=mock_repositories['pair_repository'],
            config=mock_config
        )
        return service

    @pytest.fixture
    def orders(self):
        """Create a set of orders for testing pairing operations."""
        # Create Order objects rather than mocks to test actual proximity logic
        order1 = Order("O1", [3, 4], [7, 7], 80)  # Restaurant at [3,4], customer at [7,7]
        order2 = Order("O2", [3.5, 3.5], [8, 8], 85)  # Restaurant close to order1, customer nearby
        order3 = Order("O3", [10, 10], [15, 15], 90)  # Far away from others
        order4 = Order("O4", [3, 4], [20, 20], 95)  # Same restaurant as order1, customer far
        return {
            'order1': order1,
            'order2': order2,
            'order3': order3,
            'order4': order4
        }

    # Test Group 1: _check_proximity_constraints method
    def test_check_proximity_constraints_both_within_threshold(self, service, orders):
        """Test proximity check when both restaurants and customers are within thresholds."""
        # ARRANGE
        order1 = orders['order1']  # Restaurant at [3,4], customer at [7,7]
        order2 = orders['order2']  # Restaurant at [3.5,3.5], customer at [8,8]
        # Restaurant distance ≈ 0.71, Customer distance ≈ 1.41
        # Both below thresholds (2.0 and 3.0)
        
        # ACT
        result = service._check_proximity_constraints(order1, order2)
        
        # ASSERT
        assert result is True

    def test_check_proximity_constraints_restaurants_too_far(self, service, orders):
        """Test proximity check when restaurants are too far apart."""
        # ARRANGE
        order1 = orders['order1']  # Restaurant at [3,4]
        order3 = orders['order3']  # Restaurant at [10,10]
        # Restaurant distance ≈ 9.22, exceeds threshold of 2.0
        
        # ACT
        result = service._check_proximity_constraints(order1, order3)
        
        # ASSERT
        assert result is False

    def test_check_proximity_constraints_customers_too_far(self, service, orders):
        """Test proximity check when customers are too far apart."""
        # ARRANGE
        order1 = orders['order1']  # Customer at [7,7]
        order4 = orders['order4']  # Same restaurant as order1, but customer at [20,20]
        # Customer distance ≈ 18.38, exceeds threshold of 3.0
        
        # ACT
        result = service._check_proximity_constraints(order1, order4)
        
        # ASSERT
        assert result is False

    def test_check_proximity_constraints_same_restaurant(self, service, orders):
        """Test proximity check when orders are from the exact same restaurant."""
        # ARRANGE
        order1 = orders['order1']  # Restaurant at [3,4], customer at [7,7]
        order4 = orders['order4']  # Same restaurant at [3,4], customer at [20,20]
        # Override the config for this test to allow larger customer threshold
        service.config.customers_proximity_threshold = 20.0
        
        # ACT
        result = service._check_proximity_constraints(order1, order4)
        
        # ASSERT
        assert result is True

    # Test Group 2: calculate_travel_distance method
    def test_calculate_travel_distance_simple_path(self, service):
        """Test calculation of travel distance for a simple path."""
        # ARRANGE
        # A simple path with right-angle movements:
        # [0,0] -> [3,0] -> [3,4]
        # Distance = 3 + 4 = 7
        sequence = [[0, 0], [3, 0], [3, 4]]
        
        # ACT
        distance = service.calculate_travel_distance(sequence)
        
        # ASSERT
        assert distance == pytest.approx(7.0)

    def test_calculate_travel_distance_diagonal_path(self, service):
        """Test calculation of travel distance for a path with diagonal movements."""
        # ARRANGE
        # A path with a diagonal movement:
        # [0,0] -> [3,4] -> [6,8]
        # Distance = 5 + 5 = 10
        sequence = [[0, 0], [3, 4], [6, 8]]
        
        # ACT
        distance = service.calculate_travel_distance(sequence)
        
        # ASSERT
        assert distance == pytest.approx(10.0)

    def test_calculate_travel_distance_single_point(self, service):
        """Test calculation of travel distance for a sequence with just one point."""
        # ARRANGE
        sequence = [[1, 1]]
        
        # ACT
        distance = service.calculate_travel_distance(sequence)
        
        # ASSERT
        assert distance == 0.0, "Distance for a single point should be 0"

    def test_calculate_travel_distance_empty_sequence(self, service):
        """Test calculation of travel distance for an empty sequence."""
        # ARRANGE
        sequence = []
        
        # ACT
        distance = service.calculate_travel_distance(sequence)
        
        # ASSERT
        assert distance == 0.0, "Distance for an empty sequence should be 0"

    # Test Group 3: evaluate_sequences method
    def test_evaluate_sequences_different_restaurants(self, service, orders):
        """Test sequence evaluation for orders from different restaurants."""
        # ARRANGE
        order1 = orders['order1']  # Restaurant at [3,4], customer at [7,7]
        order2 = orders['order2']  # Restaurant at [3.5,3.5], customer at [8,8]
        
        # ACT
        best_sequence, best_cost = service.evaluate_sequences(order1, order2)
        
        # ASSERT
        # There should be 4 possible sequences for different restaurants
        assert len(best_sequence) == 4, "Sequence should have 4 stops"
        # Best sequence should be the one with minimum travel distance
        assert best_cost > 0, "Best cost should be positive"

    def test_evaluate_sequences_same_restaurant(self, service):
        """Test sequence evaluation for orders from the same restaurant."""
        # ARRANGE
        # Create two orders with the same restaurant location
        order1 = Order("O1", [5, 5], [8, 8], 80)   # Restaurant at [5,5], customer at [8,8]
        order2 = Order("O2", [5, 5], [10, 10], 85) # Same restaurant, customer at [10,10]
        
        # ACT
        best_sequence, best_cost = service.evaluate_sequences(order1, order2)
        
        # ASSERT
        # There should be 3 points in the sequence for same restaurant (not 4)
        assert len(best_sequence) == 3, "Sequence should have 3 stops for same restaurant"
        # First point should be the restaurant
        assert best_sequence[0] == [5, 5], "First stop should be the restaurant"
        # Best cost should be the minimum travel distance
        assert best_cost > 0, "Best cost should be positive"

    # Test Group 4: calculate_best_match method
    def test_calculate_best_match_with_multiple_candidates(self, service, orders):
        """Test finding the best match from multiple candidates."""
        # ARRANGE
        order1 = orders['order1']
        # Multiple candidates with different distances
        candidates = [orders['order2'], orders['order4']]
        
        # Mock the evaluate_sequences method to return known values
        with patch.object(service, 'evaluate_sequences') as mock_eval:
            # Set up return values for each candidate:
            # order2: cost = 10, order4: cost = 15
            mock_eval.side_effect = [
                ([[3, 4], [3.5, 3.5], [7, 7], [8, 8]], 10.0),  # order1-order2 sequence and cost
                ([[3, 4], [7, 7], [20, 20]], 15.0)            # order1-order4 sequence and cost
            ]
            
            # ACT
            best_match, best_sequence, best_cost = service.calculate_best_match(order1, candidates)
            
            # ASSERT
            assert best_match is orders['order2'], "Should select order with lowest cost"
            assert best_sequence == [[3, 4], [3.5, 3.5], [7, 7], [8, 8]]
            assert best_cost == 10.0

    def test_calculate_best_match_with_one_candidate(self, service, orders):
        """Test finding the best match with only one candidate."""
        # ARRANGE
        order1 = orders['order1']
        candidates = [orders['order2']]  # Only one candidate
        
        # Mock the evaluate_sequences method to return a known value
        with patch.object(service, 'evaluate_sequences') as mock_eval:
            mock_eval.return_value = ([[3, 4], [3.5, 3.5], [7, 7], [8, 8]], 10.0)
            
            # ACT
            best_match, best_sequence, best_cost = service.calculate_best_match(order1, candidates)
            
            # ASSERT
            assert best_match is orders['order2'], "Should select the only candidate"
            assert best_sequence == [[3, 4], [3.5, 3.5], [7, 7], [8, 8]]
            assert best_cost == 10.0

    def test_calculate_best_match_with_no_candidates(self, service, orders):
        """Test finding the best match with no candidates."""
        # ARRANGE
        order1 = orders['order1']
        candidates = []  # No candidates
        
        # ACT
        best_match, best_sequence, best_cost = service.calculate_best_match(order1, candidates)
        
        # ASSERT
        assert best_match is None, "Best match should be None when no candidates"
        assert best_sequence is None, "Best sequence should be None when no candidates"
        assert best_cost == float('inf'), "Best cost should be infinity when no candidates"

    # Test Group 5: find_pairing_candidates method
    def test_find_pairing_candidates(self, service, orders):
        """Test finding pairing candidates for a new order."""
        # ARRANGE
        order1 = orders['order1']  # New order
        
        # Set up the order repository to return all other orders
        all_singles = [orders['order1'], orders['order2'], orders['order3'], orders['order4']]
        service.order_repository.find_unassigned_orders.return_value = all_singles
        
        # ACT
        # Spy on _check_proximity_constraints to see how it's called
        with patch.object(service, '_check_proximity_constraints') as mock_check:
            # Make constraint check return True only for order2 (close to order1)
            mock_check.side_effect = lambda o1, o2: o2.order_id == "O2"
            
            candidates = service.find_pairing_candidates(order1)
            
            # ASSERT
            # Should have called check only for orders other than order1
            assert mock_check.call_count == 3, "Should check 3 orders (excluding the new order itself)"
            
            # Only order2 should be returned as candidate
            assert len(candidates) == 1, "Should find 1 suitable candidate"
            assert candidates[0] == orders['order2'], "Order2 should be the only candidate"
            
            # Verify it didn't include the new order itself
            for call_args in mock_check.call_args_list:
                args, _ = call_args
                assert args[1] != order1, "Should not try to pair an order with itself"

    def test_find_pairing_candidates_with_no_matches(self, service, orders):
        """Test finding pairing candidates when no orders match criteria."""
        # ARRANGE
        order1 = orders['order1']  # New order
        
        # Set up the order repository to return all other orders
        all_singles = [orders['order1'], orders['order2'], orders['order3'], orders['order4']]
        service.order_repository.find_unassigned_orders.return_value = all_singles
        
        # ACT
        # Make constraint check always return False
        with patch.object(service, '_check_proximity_constraints', return_value=False):
            candidates = service.find_pairing_candidates(order1)
            
            # ASSERT
            assert len(candidates) == 0, "Should find no suitable candidates"

    def test_find_pairing_candidates_empty_repository(self, service, orders):
        """Test finding pairing candidates when repository is empty."""
        # ARRANGE
        order1 = orders['order1']  # New order
        
        # Set up the order repository to return empty list
        service.order_repository.find_unassigned_orders.return_value = []
        
        # ACT
        candidates = service.find_pairing_candidates(order1)
        
        # ASSERT
        assert len(candidates) == 0, "Should find no candidates when repository is empty"