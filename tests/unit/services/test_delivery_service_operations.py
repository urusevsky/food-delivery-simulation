# tests/unit/services/test_delivery_service_operations.py
import pytest
from unittest.mock import Mock, patch
from delivery_sim.services.delivery_service import DeliveryService
from delivery_sim.utils.location_utils import calculate_distance


class TestDeliveryServiceOperations:
    """Test suite for DeliveryService operations that can be tested in isolation."""

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
        config.driver_speed = 0.5  # km per minute
        return config

    @pytest.fixture
    def mock_repositories(self):
        """Create mock repositories."""
        driver_repository = Mock()
        order_repository = Mock()
        pair_repository = Mock()
        delivery_unit_repository = Mock()
        return {
            'driver': driver_repository,
            'order': order_repository,
            'pair': pair_repository,
            'delivery_unit': delivery_unit_repository
        }

    @pytest.fixture
    def service(self, mock_env, mock_repositories, mock_config):
        """Create a DeliveryService instance with mocked dependencies."""
        event_dispatcher = Mock()
        service = DeliveryService(
            env=mock_env,
            event_dispatcher=event_dispatcher,
            driver_repository=mock_repositories['driver'],
            order_repository=mock_repositories['order'],
            pair_repository=mock_repositories['pair'],
            delivery_unit_repository=mock_repositories['delivery_unit'],
            config=mock_config
        )
        return service

    # Test Group 1: _calculate_travel_time method
    def test_calculate_travel_time_zero_distance(self, service):
        """Test travel time calculation with zero distance."""
        # ARRANGE
        origin = [5, 5]
        destination = [5, 5]  # Same location
        
        # ACT
        travel_time = service._calculate_travel_time(origin, destination)
        
        # ASSERT
        assert travel_time == 0.0, "Travel time for same location should be 0"

    def test_calculate_travel_time_horizontal(self, service):
        """Test travel time calculation with horizontal movement."""
        # ARRANGE
        origin = [0, 0]
        destination = [10, 0]  # 10 km horizontally
        # With speed of 0.5 km/min, should take 20 minutes
        
        # ACT
        travel_time = service._calculate_travel_time(origin, destination)
        
        # ASSERT
        assert travel_time == 20.0, "Travel time should be distance / speed = 10 / 0.5 = 20 minutes"

    def test_calculate_travel_time_vertical(self, service):
        """Test travel time calculation with vertical movement."""
        # ARRANGE
        origin = [0, 0]
        destination = [0, 5]  # 5 km vertically
        # With speed of 0.5 km/min, should take 10 minutes
        
        # ACT
        travel_time = service._calculate_travel_time(origin, destination)
        
        # ASSERT
        assert travel_time == 10.0, "Travel time should be distance / speed = 5 / 0.5 = 10 minutes"

    def test_calculate_travel_time_diagonal(self, service):
        """Test travel time calculation with diagonal movement."""
        # ARRANGE
        origin = [0, 0]
        destination = [3, 4]  # 5 km diagonally (3-4-5 triangle)
        # With speed of 0.5 km/min, should take 10 minutes
        
        # ACT
        travel_time = service._calculate_travel_time(origin, destination)
        
        # ASSERT
        assert travel_time == 10.0, "Travel time should be distance / speed = 5 / 0.5 = 10 minutes"

    def test_calculate_travel_time_custom_speed(self, service):
        """Test travel time calculation with a different driver speed."""
        # ARRANGE
        origin = [0, 0]
        destination = [10, 0]  # 10 km
        # Temporarily change the driver_speed
        service.config.driver_speed = 1.0  # 1 km per minute
        
        # ACT
        travel_time = service._calculate_travel_time(origin, destination)
        
        # ASSERT
        assert travel_time == 10.0, "Travel time should be distance / speed = 10 / 1.0 = 10 minutes"
        
        # Reset speed for other tests
        service.config.driver_speed = 0.5

    def test_calculate_travel_time_with_calculate_distance(self, service):
        """Test travel time calculation using the actual distance calculation."""
        # ARRANGE
        origin = [2, 3]
        destination = [5, 7]
        distance = calculate_distance(origin, destination)  # Use actual distance calculation
        expected_time = distance / service.config.driver_speed
        
        # ACT
        travel_time = service._calculate_travel_time(origin, destination)
        
        # ASSERT
        assert travel_time == pytest.approx(expected_time), f"Travel time should be {expected_time} minutes"

    # Test Group 2: _generate_sequence_description method
    def test_generate_sequence_description_empty(self, service):
        """Test generating description for an empty sequence."""
        # ARRANGE
        sequence = []
        
        # ACT
        description = service._generate_sequence_description(sequence)
        
        # ASSERT
        assert description == "", "Description of empty sequence should be empty string"

    def test_generate_sequence_description_single_point(self, service):
        """Test generating description for a sequence with a single point."""
        # ARRANGE
        sequence = [[3.14159, 2.71828]]
        
        # ACT
        description = service._generate_sequence_description(sequence)
        
        # ASSERT
        assert description == "[3.14, 2.72]", "Should format coordinates with 2 decimal places"

    def test_generate_sequence_description_multiple_points(self, service):
        """Test generating description for a sequence with multiple points."""
        # ARRANGE
        sequence = [[0, 0], [3, 4], [6, 8]]
        
        # ACT
        description = service._generate_sequence_description(sequence)
        
        # ASSERT
        assert description == "[0.00, 0.00] -> [3.00, 4.00] -> [6.00, 8.00]", \
            "Should format coordinates with arrow separators"

    def test_generate_sequence_description_with_decimals(self, service):
        """Test generating description with decimal coordinates."""
        # ARRANGE
        sequence = [[1.2345, 6.7891], [2.3456, 7.8912]]
        
        # ACT
        description = service._generate_sequence_description(sequence)
        
        # ASSERT
        assert description == "[1.23, 6.79] -> [2.35, 7.89]", \
            "Should round coordinates to 2 decimal places"