# tests/unit/services/test_arrival_service_operations.py
import pytest
import numpy as np
from unittest.mock import Mock, patch
from delivery_sim.services.order_arrival_service import OrderArrivalService
from delivery_sim.services.driver_arrival_service import DriverArrivalService
from delivery_sim.entities.restaurant import Restaurant


class TestOrderArrivalServiceOperations:
    """Test suite for OrderArrivalService operations that can be tested in isolation."""

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
        config.mean_order_inter_arrival_time = 2.0  # minutes
        config.delivery_area_size = 10.0  # km x km
        return config

    @pytest.fixture
    def mock_operational_rng_manager(self):
        """Create a mock RNG manager."""
        rng_manager = Mock()
        
        # Create mock random streams
        order_stream = Mock()
        location_stream = Mock()
        restaurant_stream = Mock()
        
        # Configure the manager to return the mock streams
        rng_manager.get_stream.side_effect = lambda name: {
            'order_arrivals': order_stream,
            'customer_locations': location_stream,
            'restaurant_selection': restaurant_stream
        }.get(name)
        
        return {
            'manager': rng_manager,
            'order_stream': order_stream,
            'location_stream': location_stream,
            'restaurant_stream': restaurant_stream
        }

    @pytest.fixture
    def mock_restaurant_repository(self):
        """Create a mock restaurant repository with sample restaurants."""
        repository = Mock()
        
        # Create mock restaurants
        restaurant1 = Restaurant("R1", [2, 3])
        restaurant2 = Restaurant("R2", [5, 6])
        restaurant3 = Restaurant("R3", [8, 9])
        
        # Configure the repository to return these restaurants
        repository.find_all.return_value = [restaurant1, restaurant2, restaurant3]
        
        return repository

    @pytest.fixture
    def service(self, mock_env, mock_config, mock_operational_rng_manager, mock_restaurant_repository):
        """Create an OrderArrivalService instance with mocked dependencies."""
        event_dispatcher = Mock()
        order_repository = Mock()
        id_generator = Mock()
        
        service = OrderArrivalService(
            env=mock_env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repository,
            restaurant_repository=mock_restaurant_repository,
            config=mock_config,
            id_generator=id_generator,
            operational_rng_manager=mock_operational_rng_manager['manager']
        )
        return service

    # Test Group 1: _generate_inter_arrival_time method
    def test_order_generate_inter_arrival_time(self, service, mock_operational_rng_manager):
        """Test generating inter-arrival time for orders."""
        # ARRANGE
        expected_time = 2.5
        # Configure the mock to return a known value
        mock_operational_rng_manager['order_stream'].exponential.return_value = expected_time
        
        # ACT
        time = service._generate_inter_arrival_time()
        
        # ASSERT
        assert time == expected_time, f"Should return the time from the exponential distribution: {expected_time}"
        # Verify the exponential distribution was called with the correct parameter
        mock_operational_rng_manager['order_stream'].exponential.assert_called_once_with(
            service.config.mean_order_inter_arrival_time
        )

    # Test Group 2: _select_restaurant_location method
    def test_select_restaurant_location(self, service, mock_operational_rng_manager, mock_restaurant_repository):
        """Test selecting a restaurant location for a new order."""
        # ARRANGE
        # Configure the mock to select the second restaurant
        mock_operational_rng_manager['restaurant_stream'].choice.return_value = \
            mock_restaurant_repository.find_all.return_value[1]  # Restaurant2
        
        # ACT
        location = service._select_restaurant_location()
        
        # ASSERT
        assert location == [5, 6], "Should return the location of the selected restaurant"
        # Verify the choice was made from all restaurants
        mock_operational_rng_manager['restaurant_stream'].choice.assert_called_once_with(
            mock_restaurant_repository.find_all.return_value
        )

    # Test Group 3: _generate_customer_location method
    def test_generate_customer_location(self, service, mock_operational_rng_manager):
        """Test generating a customer location for a new order."""
        # ARRANGE
        # Configure the mock to return a known location
        mock_operational_rng_manager['location_stream'].uniform.return_value = np.array([3.5, 7.8])
        
        # ACT
        location = service._generate_customer_location()
        
        # ASSERT
        assert location == [3.5, 7.8], "Should return the location from the uniform distribution"
        # Verify the uniform distribution was called with the correct parameters
        mock_operational_rng_manager['location_stream'].uniform.assert_called_once_with(
            0, service.config.delivery_area_size, size=2
        )


class TestDriverArrivalServiceOperations:
    """Test suite for DriverArrivalService operations that can be tested in isolation."""

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
        config.mean_driver_inter_arrival_time = 3.0  # minutes
        config.delivery_area_size = 10.0  # km x km
        config.mean_service_duration = 120.0  # minutes
        config.service_duration_std_dev = 30.0  # minutes
        config.min_service_duration = 60.0  # minutes
        config.max_service_duration = 240.0  # minutes
        return config

    @pytest.fixture
    def mock_operational_rng_manager(self):
        """Create a mock RNG manager."""
        rng_manager = Mock()
        
        # Create mock random streams
        driver_stream = Mock()
        location_stream = Mock()
        service_stream = Mock()
        
        # Configure the manager to return the mock streams
        rng_manager.get_stream.side_effect = lambda name: {
            'driver_arrivals': driver_stream,
            'driver_initial_locations': location_stream,
            'service_duration': service_stream
        }.get(name)
        
        return {
            'manager': rng_manager,
            'driver_stream': driver_stream,
            'location_stream': location_stream,
            'service_stream': service_stream
        }

    @pytest.fixture
    def service(self, mock_env, mock_config, mock_operational_rng_manager):
        """Create a DriverArrivalService instance with mocked dependencies."""
        event_dispatcher = Mock()
        driver_repository = Mock()
        id_generator = Mock()
        
        service = DriverArrivalService(
            env=mock_env,
            event_dispatcher=event_dispatcher,
            driver_repository=driver_repository,
            config=mock_config,
            id_generator=id_generator,
            operational_rng_manager=mock_operational_rng_manager['manager']
        )
        return service

    # Test Group 1: _generate_inter_arrival_time method
    def test_driver_generate_inter_arrival_time(self, service, mock_operational_rng_manager):
        """Test generating inter-arrival time for drivers."""
        # ARRANGE
        expected_time = 3.5
        # Configure the mock to return a known value
        mock_operational_rng_manager['driver_stream'].exponential.return_value = expected_time
        
        # ACT
        time = service._generate_inter_arrival_time()
        
        # ASSERT
        assert time == expected_time, f"Should return the time from the exponential distribution: {expected_time}"
        # Verify the exponential distribution was called with the correct parameter
        mock_operational_rng_manager['driver_stream'].exponential.assert_called_once_with(
            service.config.mean_driver_inter_arrival_time
        )

    # Test Group 2: _generate_initial_location method
    def test_generate_initial_location(self, service, mock_operational_rng_manager):
        """Test generating an initial location for a new driver."""
        # ARRANGE
        # Configure the mock to return a known location
        mock_operational_rng_manager['location_stream'].uniform.return_value = np.array([4.2, 6.7])
        
        # ACT
        location = service._generate_initial_location()
        
        # ASSERT
        assert location == [4.2, 6.7], "Should return the location from the uniform distribution"
        # Verify the uniform distribution was called with the correct parameters
        mock_operational_rng_manager['location_stream'].uniform.assert_called_once_with(
            0, service.config.delivery_area_size, size=2
        )

    # Test Group 3: _generate_service_duration method
    @patch('scipy.stats.lognorm')
    def test_generate_service_duration(self, mock_lognorm, service, mock_operational_rng_manager):
        """Test generating a service duration for a new driver."""
        # ARRANGE
        # Configure the mocks
        mock_distribution = Mock()
        mock_lognorm.return_value = mock_distribution
        
        # Set up the bounds
        mock_distribution.cdf.side_effect = [0.1, 0.9]  # min and max bounds
        mock_operational_rng_manager['service_stream'].uniform.return_value = 0.5  # uniform random value
        
        # Set the ppf return value (inverse CDF)
        mock_distribution.ppf.return_value = 120.0  # 2 hours
        
        # ACT
        duration = service._generate_service_duration()
        
        # ASSERT
        # Use an approximate match instead of exact
        assert duration == pytest.approx(120.0, rel=0.05), "Should return the service duration from the lognormal distribution"