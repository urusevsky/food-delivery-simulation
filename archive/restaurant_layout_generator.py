import numpy as np
from delivery_sim.entities.restaurant import Restaurant


# NOTE: This more complex pattern-based layout generator is currently archived.
# The simulation uses simple uniform random distribution for restaurant placement.
# This generator may be utilized in future research exploring the effects of 
# different spatial patterns on system performance.

class RestaurantLayoutGenerator:
    """
    Generator for creating different spatial patterns of restaurant locations.
    
    This utility class provides methods for creating three distinct restaurant layout patterns:
    - Dispersed: Restaurants are spread evenly throughout the delivery area (maximum separation)
    - Clustered: Restaurants are grouped in districts or clusters (mimicking food courts/districts)
    - Mixed: A combination of clustered and dispersed patterns (realistic city layout)
    
    The generator ensures consistent results for the same input parameters by using
    the provided structural random number generator.
    """
    
    # Pattern type constants
    PATTERN_DISPERSED = 'dispersed'
    PATTERN_CLUSTERED = 'clustered'
    PATTERN_MIXED = 'mixed'
    
    def __init__(self, area_size, structural_rng):
        """
        Initialize the layout generator.
        
        Args:
            area_size: Size of delivery area (square with side length in km)
            structural_rng: Random number generator for structural elements
        """
        self.area_size = area_size
        self.rng = structural_rng
    
    def generate_restaurants(self, pattern_type, num_restaurants):
        """
        Generate restaurant entities with locations based on specified pattern.
        
        Args:
            pattern_type: Type of spatial pattern ('dispersed', 'clustered', or 'mixed')
            num_restaurants: Number of restaurants to generate
            
        Returns:
            list: Restaurant objects with generated locations
            
        Raises:
            ValueError: If pattern_type is not one of the supported patterns
        """
        # Generate location coordinates based on pattern
        if pattern_type == self.PATTERN_DISPERSED:
            locations = self._generate_dispersed(num_restaurants)
        elif pattern_type == self.PATTERN_CLUSTERED:
            locations = self._generate_clustered(num_restaurants)
        elif pattern_type == self.PATTERN_MIXED:
            locations = self._generate_mixed(num_restaurants)
        else:
            raise ValueError(f"Unsupported pattern type: {pattern_type}. "
                             f"Must be one of: {self.PATTERN_DISPERSED}, "
                             f"{self.PATTERN_CLUSTERED}, {self.PATTERN_MIXED}")
        
        # Create Restaurant objects from locations
        restaurants = [
            Restaurant(restaurant_id=i, location=loc) 
            for i, loc in enumerate(locations)
        ]
        
        return restaurants
    
    def _calculate_distances(self, point, existing_points):
        """
        Calculate distances from a point to all existing points.
        
        Args:
            point: The [x, y] coordinates to measure from
            existing_points: List of existing [x, y] coordinates
            
        Returns:
            array: Distances from point to each existing point
        """
        if not existing_points:
            return np.array([])
            
        return np.sqrt(np.sum((np.array(existing_points) - np.array(point))**2, axis=1))
    
    def _generate_dispersed(self, num_restaurants):
        """
        Generate a dispersed restaurant layout.
        
        This algorithm places restaurants with maximal separation by:
        1. Placing the first restaurant randomly
        2. For each subsequent restaurant:
           - Generating multiple candidate positions
           - Selecting the candidate with the greatest minimum distance to existing restaurants
        
        Args:
            num_restaurants: Number of restaurants to generate
            
        Returns:
            list: List of [x, y] coordinates for restaurant locations
        """
        # Calculate the number of candidate positions based on area density
        # More restaurants => more candidates needed for good dispersion
        density = num_restaurants / (self.area_size ** 2)
        candidate_count = min(int(10 * density * self.area_size), 30)
        
        # Initialize restaurant locations list
        locations = []
        
        # Place first restaurant randomly
        locations.append(self.rng.generate_uniform(0, self.area_size, size=2).tolist())
        
        # Place remaining restaurants with maximal separation
        for i in range(1, num_restaurants):
            candidates = [
                self.rng.generate_uniform(0, self.area_size, size=2).tolist()
                for _ in range(candidate_count)
            ]
            
            # Find candidate with maximum minimum distance to existing restaurants
            best_candidate = None
            max_min_distance = 0
            
            for candidate in candidates:
                min_distance = min(self._calculate_distances(candidate, locations))
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
            
            locations.append(best_candidate)
        
        return locations
    
    def _generate_clustered(self, num_restaurants):
        """
        Generate a clustered restaurant layout.
        
        This algorithm creates distinct restaurant clusters by:
        1. Generating a small number of well-separated cluster centers
        2. Distributing restaurants around these centers
        
        Args:
            num_restaurants: Number of restaurants to generate
            
        Returns:
            list: List of [x, y] coordinates for restaurant locations
        """
        # Determine number of clusters (more restaurants = more clusters)
        n_clusters = max(2, min(num_restaurants // 3, int(np.sqrt(num_restaurants))))
        
        # Generate cluster centers using dispersed algorithm
        cluster_centers = []
        
        # Place first center randomly
        cluster_centers.append(self.rng.generate_uniform(0, self.area_size, size=2).tolist())
        
        # Place remaining centers with maximal separation (similar to dispersed algorithm)
        for i in range(1, n_clusters):
            candidates = [
                self.rng.generate_uniform(0, self.area_size, size=2).tolist()
                for _ in range(20)  # Fixed candidate count for cluster centers
            ]
            
            best_candidate = None
            max_min_distance = 0
            
            for candidate in candidates:
                min_distance = min(self._calculate_distances(candidate, cluster_centers))
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
            
            cluster_centers.append(best_candidate)
        
        # Distribute restaurants around cluster centers
        locations = []
        cluster_radius = self.area_size / (n_clusters * 2)  # Radius based on number of clusters
        
        for i in range(num_restaurants):
            # Select a cluster center (cycling through centers)
            center = cluster_centers[i % n_clusters]
            
            # Generate random angle and distance from center
            angle = self.rng.generate_uniform(0, 2 * np.pi)
            radius = self.rng.generate_uniform(0, cluster_radius)
            
            # Calculate position
            pos = [
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle)
            ]
            
            # Ensure position is within bounds
            pos[0] = min(max(pos[0], 0), self.area_size)
            pos[1] = min(max(pos[1], 0), self.area_size)
            
            locations.append(pos)
        
        return locations
    
    def _generate_mixed(self, num_restaurants):
        """
        Generate a mixed restaurant layout.
        
        This algorithm creates a realistic city-like pattern by:
        1. Placing half the restaurants in clusters (restaurant districts)
        2. Placing the other half in a dispersed pattern (standalone restaurants)
        
        Args:
            num_restaurants: Number of restaurants to generate
            
        Returns:
            list: List of [x, y] coordinates for restaurant locations
        """
        # Calculate how many restaurants to place with each method
        clustered_count = num_restaurants // 2
        dispersed_count = num_restaurants - clustered_count
        
        # Generate clustered restaurants
        locations = []
        
        if clustered_count > 0:
            # Determine number of clusters
            n_clusters = max(2, min(clustered_count // 3, int(np.sqrt(clustered_count))))
            
            # Generate cluster centers
            cluster_centers = []
            cluster_centers.append(self.rng.generate_uniform(0, self.area_size, size=2).tolist())
            
            for i in range(1, n_clusters):
                candidates = [
                    self.rng.generate_uniform(0, self.area_size, size=2).tolist()
                    for _ in range(20)
                ]
                
                best_candidate = None
                max_min_distance = 0
                
                for candidate in candidates:
                    min_distance = min(self._calculate_distances(candidate, cluster_centers))
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_candidate = candidate
                
                cluster_centers.append(best_candidate)
            
            # Distribute clustered restaurants
            cluster_radius = self.area_size / (n_clusters * 2)
            
            for i in range(clustered_count):
                center = cluster_centers[i % n_clusters]
                angle = self.rng.generate_uniform(0, 2 * np.pi)
                radius = self.rng.generate_uniform(0, cluster_radius)
                
                pos = [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle)
                ]
                
                pos[0] = min(max(pos[0], 0), self.area_size)
                pos[1] = min(max(pos[1], 0), self.area_size)
                
                locations.append(pos)
        
        # Add dispersed restaurants
        if dispersed_count > 0:
            candidate_count = 20  # Fixed candidate count for dispersed points
            
            for i in range(dispersed_count):
                candidates = [
                    self.rng.generate_uniform(0, self.area_size, size=2).tolist()
                    for _ in range(candidate_count)
                ]
                
                best_candidate = None
                max_min_distance = 0
                
                for candidate in candidates:
                    # If no locations yet, any candidate is fine
                    if not locations:
                        min_distance = float('inf')
                    else:
                        min_distance = min(self._calculate_distances(candidate, locations))
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_candidate = candidate
                
                locations.append(best_candidate)
        
        return locations