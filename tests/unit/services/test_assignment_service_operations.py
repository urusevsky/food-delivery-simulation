# tests/unit/services/test_assignment_service_operations.py
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
    """Test suite for AssignmentService operations that can be tested in isolation."""

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
        config.throughput_factor = 1.5  # km per additional order
        config.age_factor = 0.1  # km per minute waiting
        config.immediate_assignment_threshold = 2.0  # km in adjusted cost
        config.driver_speed = 0.5  # km per minute
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
    def service(self, mock_env, mock_repositories, mock_config):
        """Create an AssignmentService instance with mocked dependencies."""
        event_dispatcher = Mock()
        service = AssignmentService(
            env=mock_env,
            event_dispatcher=event_dispatcher,
            order_repository=mock_repositories['order'],
            driver_repository=mock_repositories['driver'],
            pair_repository=mock_repositories['pair'],
            delivery_unit_repository=mock_repositories['delivery_unit'],
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

    # Test Group 1: calculate_adjusted_cost method
    def test_calculate_adjusted_cost_for_single_order(self, service, sample_driver, sample_order):
        """Test adjusted cost calculation for a single order."""
        # ARRANGE
        # Driver at [0,0], restaurant at [3,4], customer at [6,8]
        # Base cost = 5 (driver to restaurant) + 5 (restaurant to customer) = 10
        # 1 order, so throughput_component = 1.5 * 1 = 1.5
        # Order age = 20 minutes, so age_discount = 0.1 * 20 = 2.0
        # Adjusted cost = 10 - 1.5 - 2.0 = 6.5
        
        # ACT
        adjusted_cost, components = service.calculate_adjusted_cost(sample_driver, sample_order)
        
        # ASSERT
        assert components["base_cost"] == pytest.approx(10.0)
        assert components["num_orders"] == 1
        assert components["throughput_component"] == pytest.approx(1.5)
        assert components["age_minutes"] == pytest.approx(20.0)
        assert components["age_discount"] == pytest.approx(2.0)
        assert adjusted_cost == pytest.approx(6.5)

    def test_calculate_adjusted_cost_for_pair(self, service, sample_driver, sample_pair):
        """Test adjusted cost calculation for a pair of orders."""
        # ARRANGE
        # Driver at [0,0], first stop at [3,4], with remaining delivery cost = 10
        # Base cost = 5 (driver to first stop) + 10 (remaining sequence) = 15
        # 2 orders, so throughput_component = 1.5 * 2 = 3.0
        # Pair age from older order = 100 - 75 = 25 minutes
        # Age discount = 0.1 * 25 = 2.5
        # Adjusted cost = 15 - 3.0 - 2.5 = 9.5
        
        # ACT
        adjusted_cost, components = service.calculate_adjusted_cost(sample_driver, sample_pair)
        
        # ASSERT
        assert components["base_cost"] == pytest.approx(15.0)
        assert components["num_orders"] == 2
        assert components["throughput_component"] == pytest.approx(3.0)
        assert components["age_minutes"] == pytest.approx(25.0)
        assert components["age_discount"] == pytest.approx(2.5)
        assert adjusted_cost == pytest.approx(9.5)

    # Test Group 2: calculate_base_delivery_cost method
    def test_calculate_base_delivery_cost_for_order(self, service, sample_driver, sample_order):
        """Test base delivery cost calculation for a single order."""
        # ARRANGE
        # Driver at [0,0], restaurant at [3,4], customer at [6,8]
        # Distance from driver to restaurant = 5
        # Distance from restaurant to customer = 5
        # Total base cost = 10
        
        # ACT
        cost = service.calculate_base_delivery_cost(sample_driver, sample_order)
        
        # ASSERT
        assert cost == pytest.approx(10.0)

    def test_calculate_base_delivery_cost_for_pair(self, service, sample_driver, sample_pair):
        """Test base delivery cost calculation for a pair."""
        # ARRANGE
        # Driver at [0,0], first stop at [3,4]
        # Distance from driver to first stop = 5
        # Remaining sequence cost = 10 (from optimal_cost)
        # Total base cost = 15
        
        # ACT
        cost = service.calculate_base_delivery_cost(sample_driver, sample_pair)
        
        # ASSERT
        assert cost == pytest.approx(15.0)

    # Test Group 3: _find_best_match method
    def test_find_best_match_from_drivers(self, service, sample_order):
        """Test finding the best driver for an order."""
        # ARRANGE
        # Create 3 potential drivers with different locations
        driver1 = Mock(spec=Driver)
        driver1.driver_id = "D1"
        driver1.location = [0, 0]  # 5 units from restaurant
        driver1.entity_type = EntityType.DRIVER
        
        driver2 = Mock(spec=Driver)
        driver2.driver_id = "D2"
        driver2.location = [2, 2]  # 3 units from restaurant
        driver2.entity_type = EntityType.DRIVER
        
        driver3 = Mock(spec=Driver)
        driver3.driver_id = "D3"
        driver3.location = [3, 3]  # 1 unit from restaurant
        driver3.entity_type = EntityType.DRIVER
        
        candidates = [driver1, driver2, driver3]
        
        # Mock the calculate_adjusted_cost method to return known values
        # Driver1: 10, Driver2: 8, Driver3: 6
        with patch.object(service, 'calculate_adjusted_cost') as mock_calc:
            mock_calc.side_effect = [
                (10.0, {"base_cost": 15.0}),
                (8.0, {"base_cost": 13.0}),
                (6.0, {"base_cost": 11.0})
            ]
            
            # ACT
            best_match, best_cost, best_components = service._find_best_match(sample_order, candidates)
            
            # ASSERT
            assert best_match is driver3
            assert best_cost == pytest.approx(6.0)
            assert best_components["base_cost"] == pytest.approx(11.0)

    def test_find_best_match_from_delivery_entities(self, service, sample_driver):
        """Test finding the best delivery entity for a driver."""
        # ARRANGE
        # Create 2 orders and 1 pair with different locations
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
        
        # Mock the calculate_adjusted_cost method to return known values
        # Order1: 9, Order2: 7, Pair: 8
        with patch.object(service, 'calculate_adjusted_cost') as mock_calc:
            mock_calc.side_effect = [
                (9.0, {"base_cost": 12.0}),
                (7.0, {"base_cost": 10.0}),
                (8.0, {"base_cost": 15.0})
            ]
            
            # ACT
            best_match, best_cost, best_components = service._find_best_match(sample_driver, candidates)
            
            # ASSERT
            assert best_match is order2
            assert best_cost == pytest.approx(7.0)
            assert best_components["base_cost"] == pytest.approx(10.0)

    def test_find_best_match_empty_candidates(self, service, sample_driver):
        """Test finding the best match when there are no candidates."""
        # ARRANGE
        candidates = []
        
        # ACT
        best_match, best_cost, best_components = service._find_best_match(sample_driver, candidates)
        
        # ASSERT
        assert best_match is None
        assert best_cost == float('inf')
        assert best_components is None

    # Test Group 4: _generate_cost_matrix method
    def test_generate_cost_matrix(self, service):
        """Test generating cost matrix for optimization."""
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
        
        # Mock the calculate_adjusted_cost method to return known values
        with patch.object(service, 'calculate_adjusted_cost') as mock_calc:
            mock_calc.side_effect = [
                (5.0, {}),  # order1-driver1
                (6.0, {}),  # order1-driver2
                (7.0, {}),  # pair1-driver1
                (4.0, {})   # pair1-driver2
            ]
            
            # ACT
            cost_matrix = service._generate_cost_matrix(waiting_entities, available_drivers)
            
            # ASSERT
            assert len(cost_matrix) == 2  # Two waiting entities
            assert len(cost_matrix[0]) == 2  # Two drivers
            assert cost_matrix[0][0] == pytest.approx(5.0)  # order1-driver1
            assert cost_matrix[0][1] == pytest.approx(6.0)  # order1-driver2
            assert cost_matrix[1][0] == pytest.approx(7.0)  # pair1-driver1
            assert cost_matrix[1][1] == pytest.approx(4.0)  # pair1-driver2

    def test_generate_cost_matrix_single_entity_multiple_drivers(self, service):
        """Test generating cost matrix with one entity and multiple drivers."""
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
        
        # Mock the calculate_adjusted_cost method to return known values
        with patch.object(service, 'calculate_adjusted_cost') as mock_calc:
            mock_calc.side_effect = [
                (5.0, {}),  # order-driver1
                (3.0, {}),  # order-driver2
                (7.0, {})   # order-driver3
            ]
            
            # ACT
            cost_matrix = service._generate_cost_matrix(waiting_entities, available_drivers)
            
            # ASSERT
            assert len(cost_matrix) == 1  # One waiting entity
            assert len(cost_matrix[0]) == 3  # Three drivers
            assert cost_matrix[0][0] == pytest.approx(5.0)  # order-driver1
            assert cost_matrix[0][1] == pytest.approx(3.0)  # order-driver2
            assert cost_matrix[0][2] == pytest.approx(7.0)  # order-driver3

    def test_generate_cost_matrix_multiple_entities_single_driver(self, service):
        """Test generating cost matrix with multiple entities and one driver."""
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
        
        # Mock the calculate_adjusted_cost method to return known values
        with patch.object(service, 'calculate_adjusted_cost') as mock_calc:
            mock_calc.side_effect = [
                (5.0, {}),  # order1-driver
                (3.0, {}),  # order2-driver
                (7.0, {})   # pair1-driver
            ]
            
            # ACT
            cost_matrix = service._generate_cost_matrix(waiting_entities, available_drivers)
            
            # ASSERT
            assert len(cost_matrix) == 3  # Three waiting entities
            assert len(cost_matrix[0]) == 1  # One driver
            assert cost_matrix[0][0] == pytest.approx(5.0)  # order1-driver
            assert cost_matrix[1][0] == pytest.approx(3.0)  # order2-driver
            assert cost_matrix[2][0] == pytest.approx(7.0)  # pair1-driver