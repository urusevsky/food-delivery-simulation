# tests/unit/utils/test_infrastructure_analysis.py
"""
Unit tests for Infrastructure Analysis Module

This module tests the infrastructure analysis functionality that performs
Monte Carlo sampling for typical distances and other infrastructure characteristics.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from delivery_sim.utils.infrastructure_analysis import TypicalDistanceCalculator, analyze_infrastructure
from delivery_sim.entities.restaurant import Restaurant


class TestTypicalDistanceCalculator:
    """Test suite for the TypicalDistanceCalculator class."""
    
    @pytest.fixture
    def mock_restaurant_repository(self):
        """Create a mock restaurant repository with sample restaurants."""
        repo = Mock()
        
        # Create sample restaurants in a 10x10 area
        restaurants = [
            Restaurant("R1", [2, 2]),
            Restaurant("R2", [8, 2]),
            Restaurant("R3", [5, 8]),
            Restaurant("R4", [1, 9])
        ]
        repo.find_all.return_value = restaurants
        return repo
    
    @pytest.fixture
    def mock_structural_rng(self):
        """Create a mock structural RNG manager."""
        rng_manager = Mock()
        
        # Create a mock numpy random generator
        mock_rng = Mock()
        rng_manager.rng = mock_rng
        
        return rng_manager
    
    # Test Group 1: Basic functionality tests
    @patch('delivery_sim.utils.infrastructure_analysis.calculate_distance')
    def test_calculate_typical_distance_basic(self, mock_calculate_distance, mock_restaurant_repository, mock_structural_rng):
        """Test basic typical distance calculation with known inputs."""
        # ARRANGE
        area_size = 10
        sample_size = 4  # Small sample for controlled testing
        
        # Create mock numpy arrays that have .tolist() method
        mock_driver_locs = [
            Mock(tolist=Mock(return_value=[0, 0])),
            Mock(tolist=Mock(return_value=[5, 5])),
            Mock(tolist=Mock(return_value=[10, 10])),
            Mock(tolist=Mock(return_value=[2, 8]))
        ]
        mock_customer_locs = [
            Mock(tolist=Mock(return_value=[1, 1])),
            Mock(tolist=Mock(return_value=[6, 6])),
            Mock(tolist=Mock(return_value=[9, 9])),
            Mock(tolist=Mock(return_value=[3, 7]))
        ]
        
        # Set up uniform calls: driver_loc, customer_loc for each sample
        uniform_calls = []
        for i in range(sample_size):
            uniform_calls.append(mock_driver_locs[i])
            uniform_calls.append(mock_customer_locs[i])
        
        mock_structural_rng.rng.uniform.side_effect = uniform_calls
        
        # Mock restaurant selection to cycle through restaurants
        restaurants = mock_restaurant_repository.find_all.return_value
        mock_structural_rng.rng.choice.side_effect = [
            restaurants[0], restaurants[1], restaurants[2], restaurants[3]
        ]
        
        # Mock distance calculations to return known values
        # For each sample: driver->restaurant + restaurant->customer
        mock_calculate_distance.side_effect = [
            2.0, 1.5,  # Sample 1: 2.0 + 1.5 = 3.5
            3.0, 2.0,  # Sample 2: 3.0 + 2.0 = 5.0
            5.0, 1.5,  # Sample 3: 5.0 + 1.5 = 6.5
            4.0, 2.5   # Sample 4: 4.0 + 2.5 = 6.5
        ]
        
        # Expected samples: [3.5, 5.0, 6.5, 6.5]
        # Median of [3.5, 5.0, 6.5, 6.5] = 5.75
        
        # ACT
        typical_distance = TypicalDistanceCalculator.calculate_typical_distance(
            mock_restaurant_repository, area_size, mock_structural_rng, sample_size
        )
        
        # ASSERT
        assert typical_distance == pytest.approx(5.75)
        assert mock_calculate_distance.call_count == 8  # 2 calls per sample, 4 samples

    def test_calculate_typical_distance_single_restaurant(self, mock_structural_rng):
        """Test typical distance calculation with only one restaurant."""
        # ARRANGE
        repo = Mock()
        repo.find_all.return_value = [Restaurant("R1", [5, 5])]
        
        area_size = 10
        sample_size = 2
        
        # Mock numpy arrays with .tolist() method
        mock_arrays = [
            Mock(tolist=Mock(return_value=[0, 0])),  # Driver 1
            Mock(tolist=Mock(return_value=[2, 2])),  # Customer 1
            Mock(tolist=Mock(return_value=[10, 10])), # Driver 2
            Mock(tolist=Mock(return_value=[8, 8]))   # Customer 2
        ]
        
        mock_structural_rng.rng.uniform.side_effect = mock_arrays
        mock_structural_rng.rng.choice.side_effect = [
            repo.find_all.return_value[0],  # Same restaurant selected twice
            repo.find_all.return_value[0]
        ]
        
        # ACT & ASSERT - Should not raise any errors
        with patch('delivery_sim.utils.infrastructure_analysis.calculate_distance') as mock_calc:
            mock_calc.side_effect = [7.07, 4.24, 7.07, 4.24]  # Mock distances
            typical_distance = TypicalDistanceCalculator.calculate_typical_distance(
                repo, area_size, mock_structural_rng, sample_size
            )
            assert isinstance(typical_distance, float)
            assert typical_distance > 0

    def test_calculate_typical_distance_no_restaurants_raises_error(self, mock_structural_rng):
        """Test that empty restaurant repository raises appropriate error."""
        # ARRANGE
        repo = Mock()
        repo.find_all.return_value = []
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match="No restaurants found"):
            TypicalDistanceCalculator.calculate_typical_distance(
                repo, 10, mock_structural_rng, 100
            )

    @patch('delivery_sim.utils.infrastructure_analysis.np.median')
    def test_calculate_typical_distance_uses_median(self, mock_median, mock_restaurant_repository, mock_structural_rng):
        """Test that median is used for robustness against outliers."""
        # ARRANGE
        mock_median.return_value = 5.5
        area_size = 10
        sample_size = 10
        
        # Mock numpy array with .tolist() method
        mock_array = Mock(tolist=Mock(return_value=[5, 5]))
        mock_structural_rng.rng.uniform.return_value = mock_array
        mock_structural_rng.rng.choice.return_value = mock_restaurant_repository.find_all.return_value[0]
        
        # ACT
        with patch('delivery_sim.utils.infrastructure_analysis.calculate_distance', return_value=5.0):
            typical_distance = TypicalDistanceCalculator.calculate_typical_distance(
                mock_restaurant_repository, area_size, mock_structural_rng, sample_size
            )
        
        # ASSERT
        mock_median.assert_called_once()
        assert typical_distance == 5.5

    # Test Group 2: Parameter validation and edge cases
    def test_calculate_typical_distance_large_sample_size(self, mock_restaurant_repository, mock_structural_rng):
        """Test typical distance calculation with large sample size."""
        # ARRANGE
        area_size = 10
        sample_size = 1000  # Large sample size
        
        # Mock numpy array with .tolist() method
        mock_array = Mock(tolist=Mock(return_value=[5, 5]))
        mock_structural_rng.rng.uniform.return_value = mock_array
        mock_structural_rng.rng.choice.return_value = mock_restaurant_repository.find_all.return_value[0]
        
        # ACT
        with patch('delivery_sim.utils.infrastructure_analysis.calculate_distance', return_value=5.0):
            typical_distance = TypicalDistanceCalculator.calculate_typical_distance(
                mock_restaurant_repository, area_size, mock_structural_rng, sample_size
            )
        
        # ASSERT
        assert isinstance(typical_distance, float)
        assert typical_distance > 0

    def test_calculate_typical_distance_small_area(self, mock_restaurant_repository, mock_structural_rng):
        """Test typical distance calculation with small delivery area."""
        # ARRANGE
        area_size = 1  # Very small area
        sample_size = 10
        
        # Mock numpy array with .tolist() method
        mock_array = Mock(tolist=Mock(return_value=[0.5, 0.5]))
        mock_structural_rng.rng.uniform.return_value = mock_array
        mock_structural_rng.rng.choice.return_value = mock_restaurant_repository.find_all.return_value[0]
        
        # ACT
        with patch('delivery_sim.utils.infrastructure_analysis.calculate_distance', return_value=1.0):
            typical_distance = TypicalDistanceCalculator.calculate_typical_distance(
                mock_restaurant_repository, area_size, mock_structural_rng, sample_size
            )
        
        # ASSERT
        assert isinstance(typical_distance, float)
        assert typical_distance > 0


class TestAnalyzeInfrastructure:
    """Test suite for the analyze_infrastructure function."""
    
    @pytest.fixture
    def mock_restaurant_repository(self):
        """Create a mock restaurant repository."""
        repo = Mock()
        restaurants = [
            Restaurant("R1", [2, 2]),
            Restaurant("R2", [8, 2]),
            Restaurant("R3", [5, 8])
        ]
        repo.find_all.return_value = restaurants
        return repo
    
    @pytest.fixture
    def mock_structural_config(self):
        """Create a mock structural configuration."""
        config = Mock()
        config.delivery_area_size = 10
        return config
    
    @pytest.fixture
    def mock_structural_rng(self):
        """Create a mock structural RNG manager."""
        rng_manager = Mock()
        mock_rng = Mock()
        rng_manager.rng = mock_rng
        return rng_manager
    
    @pytest.fixture
    def mock_scoring_config(self):
        """Create a mock scoring configuration."""
        config = Mock()
        config.typical_distance_samples = 500
        return config
    
    # Test Group 1: Basic infrastructure analysis tests
    @patch('delivery_sim.utils.infrastructure_analysis.TypicalDistanceCalculator.calculate_typical_distance')
    def test_analyze_infrastructure_basic(self, mock_calculate_typical_distance, 
                                         mock_restaurant_repository, mock_structural_config, 
                                         mock_structural_rng, mock_scoring_config):
        """Test basic infrastructure analysis functionality."""
        # ARRANGE
        mock_calculate_typical_distance.return_value = 6.5
        
        # ACT
        result = analyze_infrastructure(
            mock_restaurant_repository, mock_structural_config, 
            mock_structural_rng, mock_scoring_config
        )
        
        # ASSERT
        assert isinstance(result, dict)
        assert result['area_size'] == 10
        assert result['restaurant_count'] == 3
        assert result['typical_distance'] == 6.5
        assert result['restaurant_density'] == pytest.approx(0.03)  # 3 / (10^2)
        assert result['average_restaurant_spacing'] == pytest.approx(5.77, abs=0.01)  # sqrt(100/3)
        assert result['analysis_sample_size'] == 500

    @patch('delivery_sim.utils.infrastructure_analysis.TypicalDistanceCalculator.calculate_typical_distance')
    def test_analyze_infrastructure_without_scoring_config(self, mock_calculate_typical_distance,
                                                           mock_restaurant_repository, mock_structural_config,
                                                           mock_structural_rng):
        """Test infrastructure analysis without scoring config (uses default sample size)."""
        # ARRANGE
        mock_calculate_typical_distance.return_value = 5.0
        
        # ACT
        result = analyze_infrastructure(
            mock_restaurant_repository, mock_structural_config, mock_structural_rng
        )
        
        # ASSERT
        assert result['analysis_sample_size'] == 1000  # Default sample size
        assert 'typical_distance' in result
        mock_calculate_typical_distance.assert_called_once()

    def test_analyze_infrastructure_no_restaurants_raises_error(self, mock_structural_config, mock_structural_rng):
        """Test that infrastructure analysis fails gracefully with no restaurants."""
        # ARRANGE
        repo = Mock()
        repo.find_all.return_value = []
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match="Cannot analyze infrastructure with no restaurants"):
            analyze_infrastructure(repo, mock_structural_config, mock_structural_rng)

    @patch('delivery_sim.utils.infrastructure_analysis.TypicalDistanceCalculator.calculate_typical_distance')
    def test_analyze_infrastructure_single_restaurant(self, mock_calculate_typical_distance,
                                                     mock_structural_config, mock_structural_rng):
        """Test infrastructure analysis with single restaurant."""
        # ARRANGE
        repo = Mock()
        repo.find_all.return_value = [Restaurant("R1", [5, 5])]
        mock_calculate_typical_distance.return_value = 4.0
        
        # ACT
        result = analyze_infrastructure(repo, mock_structural_config, mock_structural_rng)
        
        # ASSERT
        assert result['restaurant_count'] == 1
        assert result['restaurant_density'] == pytest.approx(0.01)  # 1 / (10^2)
        assert result['average_restaurant_spacing'] == pytest.approx(10.0)  # sqrt(100/1)

    # Test Group 2: Calculated metrics tests
    @patch('delivery_sim.utils.infrastructure_analysis.TypicalDistanceCalculator.calculate_typical_distance')
    def test_analyze_infrastructure_density_calculation(self, mock_calculate_typical_distance,
                                                       mock_structural_config, mock_structural_rng):
        """Test that restaurant density is calculated correctly."""
        # ARRANGE
        repo = Mock()
        repo.find_all.return_value = [Restaurant(f"R{i}", [i, i]) for i in range(5)]  # 5 restaurants
        mock_structural_config.delivery_area_size = 20  # 20x20 area = 400 km²
        mock_calculate_typical_distance.return_value = 8.0
        
        # ACT
        result = analyze_infrastructure(repo, mock_structural_config, mock_structural_rng)
        
        # ASSERT
        expected_density = 5 / (20 * 20)  # 5 restaurants / 400 km²
        assert result['restaurant_density'] == pytest.approx(expected_density)

    @patch('delivery_sim.utils.infrastructure_analysis.TypicalDistanceCalculator.calculate_typical_distance')
    def test_analyze_infrastructure_spacing_calculation(self, mock_calculate_typical_distance,
                                                       mock_structural_config, mock_structural_rng):
        """Test that average restaurant spacing is calculated correctly."""
        # ARRANGE
        repo = Mock()
        repo.find_all.return_value = [Restaurant(f"R{i}", [i, i]) for i in range(4)]  # 4 restaurants
        mock_structural_config.delivery_area_size = 8  # 8x8 area = 64 km²
        mock_calculate_typical_distance.return_value = 6.0
        
        # ACT
        result = analyze_infrastructure(repo, mock_structural_config, mock_structural_rng)
        
        # ASSERT
        expected_spacing = (64 / 4) ** 0.5  # sqrt(area_per_restaurant)
        assert result['average_restaurant_spacing'] == pytest.approx(expected_spacing)

    # Test Group 3: Integration with TypicalDistanceCalculator
    @patch('delivery_sim.utils.infrastructure_analysis.TypicalDistanceCalculator.calculate_typical_distance')
    def test_analyze_infrastructure_passes_correct_parameters(self, mock_calculate_typical_distance,
                                                             mock_restaurant_repository, mock_structural_config,
                                                             mock_structural_rng, mock_scoring_config):
        """Test that analyze_infrastructure passes correct parameters to TypicalDistanceCalculator."""
        # ARRANGE
        mock_calculate_typical_distance.return_value = 7.0
        
        # ACT
        analyze_infrastructure(
            mock_restaurant_repository, mock_structural_config, 
            mock_structural_rng, mock_scoring_config
        )
        
        # ASSERT
        mock_calculate_typical_distance.assert_called_once_with(
            restaurant_repository=mock_restaurant_repository,
            area_size=mock_structural_config.delivery_area_size,
            structural_rng=mock_structural_rng,
            sample_size=mock_scoring_config.typical_distance_samples
        )

    # Test Group 4: Edge cases and robustness tests  
    @patch('delivery_sim.utils.infrastructure_analysis.TypicalDistanceCalculator.calculate_typical_distance')
    def test_analyze_infrastructure_large_area_many_restaurants(self, mock_calculate_typical_distance,
                                                               mock_structural_config, mock_structural_rng):
        """Test infrastructure analysis with large area and many restaurants."""
        # ARRANGE
        repo = Mock()
        repo.find_all.return_value = [Restaurant(f"R{i}", [i, i]) for i in range(100)]  # 100 restaurants
        mock_structural_config.delivery_area_size = 50  # Large area
        mock_calculate_typical_distance.return_value = 15.0
        
        # ACT
        result = analyze_infrastructure(repo, mock_structural_config, mock_structural_rng)
        
        # ASSERT
        assert result['restaurant_count'] == 100
        assert result['area_size'] == 50
        assert result['restaurant_density'] == pytest.approx(0.04)  # 100 / (50^2)
        assert result['average_restaurant_spacing'] == pytest.approx(5.0)  # sqrt(2500/100)
        assert result['typical_distance'] == 15.0

    @patch('delivery_sim.utils.infrastructure_analysis.TypicalDistanceCalculator.calculate_typical_distance')
    def test_analyze_infrastructure_very_dense_configuration(self, mock_calculate_typical_distance,
                                                            mock_structural_config, mock_structural_rng):
        """Test infrastructure analysis with very dense restaurant configuration."""
        # ARRANGE
        repo = Mock()
        repo.find_all.return_value = [Restaurant(f"R{i}", [i/10, i/10]) for i in range(50)]  # Dense restaurants
        mock_structural_config.delivery_area_size = 5  # Small area
        mock_calculate_typical_distance.return_value = 2.0  # Short distances expected
        
        # ACT
        result = analyze_infrastructure(repo, mock_structural_config, mock_structural_rng)
        
        # ASSERT
        assert result['restaurant_density'] == pytest.approx(2.0)  # 50 / (5^2)
        assert result['average_restaurant_spacing'] == pytest.approx(0.71, abs=0.01)  # sqrt(25/50)
        assert result['typical_distance'] == 2.0

    # Test Group 5: Logging integration tests (if needed)
    @patch('delivery_sim.utils.infrastructure_analysis.get_logger')
    @patch('delivery_sim.utils.infrastructure_analysis.TypicalDistanceCalculator.calculate_typical_distance')
    def test_analyze_infrastructure_logging_integration(self, mock_calculate_typical_distance, mock_get_logger,
                                                       mock_restaurant_repository, mock_structural_config,
                                                       mock_structural_rng):
        """Test that analyze_infrastructure integrates properly with logging system."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_calculate_typical_distance.return_value = 5.5
        
        # ACT
        result = analyze_infrastructure(
            mock_restaurant_repository, mock_structural_config, mock_structural_rng
        )
        
        # ASSERT
        mock_get_logger.assert_called_once_with("utils.infrastructure_analyzer")
        assert mock_logger.info.call_count >= 2  # At least start and completion messages
        assert mock_logger.debug.call_count >= 1  # At least some debug messages