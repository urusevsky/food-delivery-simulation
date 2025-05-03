import simpy
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.simulation.rng_manager import StructuralRNGManager, OperationalRNGManager
from delivery_sim.entities.restaurant import Restaurant
from delivery_sim.repositories.restaurant_repository import RestaurantRepository
from delivery_sim.repositories.order_repository import OrderRepository
from delivery_sim.repositories.driver_repository import DriverRepository
from delivery_sim.repositories.pair_repository import PairRepository
from delivery_sim.repositories.delivery_unit_repository import DeliveryUnitRepository
from delivery_sim.services.order_arrival_service import OrderArrivalService
from delivery_sim.services.driver_arrival_service import DriverArrivalService
from delivery_sim.services.pairing_service import PairingService
from delivery_sim.services.assignment_service import AssignmentService
from delivery_sim.services.delivery_service import DeliveryService
from delivery_sim.services.driver_scheduling_service import DriverSchedulingService
from delivery_sim.utils.id_generator import SequentialIdGenerator


class FlatConfig:
    """
    A wrapper class that presents a flat interface for configuration attributes.
    
    This allows services to access attributes without knowing which config 
    object they belong to, simplifying service implementation.
    """
    def __init__(self, structural_config, operational_config, experiment_config):
        self.structural_config = structural_config
        self.operational_config = operational_config
        self.experiment_config = experiment_config
    
    def __getattr__(self, name):
        # Try to find the attribute in each config object
        if hasattr(self.operational_config, name):
            return getattr(self.operational_config, name)
        elif hasattr(self.structural_config, name):
            return getattr(self.structural_config, name)
        elif hasattr(self.experiment_config, name):
            return getattr(self.experiment_config, name)
        else:
            raise AttributeError(f"'FlatConfig' object has no attribute '{name}'")


class SimulationRunner:
    """
    Orchestrates the initialization and execution of the food delivery simulation.
    
    This class handles setting up all necessary components, connecting them
    appropriately, and running the simulation for the specified duration.
    """
    def __init__(self, config):
        # Store the hierarchical configuration
        self.config = config
        
        # Create a flat config wrapper for services
        self.flat_config = FlatConfig(
            config.structural_config,
            config.operational_config,
            config.experiment_config
        )
        
        # Structural component containers (constant across replications)
        self.id_generators = {}
        self.structural_rng = None
        self.restaurant_repository = None  # This will be preserved across replications
        
        # Operational component containers (reset for each replication)
        self.env = None
        self.event_dispatcher = None
        self.operational_rng = None
        self.repositories = {}
        self.services = {}
        
    def initialize(self):
        """
        Initialize all simulation components for a single simulation run.
        
        This method initializes both structural components (constant across
        replications) and operational components (specific to this run).
        
        For multiple replications, use initialize_structural_components() once
        followed by initialize_operational_components() for each replication.
        
        Returns:
            SimulationRunner: self for method chaining
        """
        # First initialize structural components
        self.initialize_structural_components()
        
        # Then initialize operational components
        self.initialize_operational_components()
        
        print("Simulation initialization complete.")
        return self
    
    def initialize_structural_components(self):
        """Initialize components that remain constant across replications."""
        # Initialize ID generators
        self.id_generators = self._setup_id_generators()
        
        # Initialize structural random number generator
        self.structural_rng = self._setup_structural_rng()
        
        # Initialize restaurant repository - this creates the fixed 
        # geographical infrastructure that remains constant across replications
        self.restaurant_repository = RestaurantRepository()
        
        # Generate restaurant infrastructure - this creates the actual
        # restaurant entities at their geographic locations
        self._setup_restaurant_infrastructure()
        
        print("Structural components initialized.")
        return self
    
    def initialize_operational_components(self, replication_number=0):
        """Initialize components that change with each replication."""
        # Create fresh SimPy environment
        self.env = simpy.Environment()
        
        # Create fresh event dispatcher
        self.event_dispatcher = EventDispatcher()
        
        # Set up operational random number generator
        self.operational_rng = self._setup_operational_rng(replication_number)
        
        # Initialize repositories (except restaurant which is preserved)
        self.repositories = self._setup_repositories()
        
        # Initialize services
        self.services = self._setup_services()
        
        print(f"Operational components initialized for replication {replication_number}.")
        return self
    
    def run(self):
        """Run the simulation for the configured duration."""
        print(f"Starting simulation run for {self.config.experiment_config.simulation_duration} minutes")
        
        # Run the simulation
        self.env.run(until=self.config.experiment_config.simulation_duration)
        
        print(f"Simulation completed")
        
        return self.repositories
    
    def _setup_id_generators(self):
        """Set up ID generators for each entity type."""
        return {
            'order': SequentialIdGenerator(1),
            'driver': SequentialIdGenerator(1),
            'restaurant': SequentialIdGenerator(1)
        }
    
    def _setup_structural_rng(self):
        """Set up the random number generator for structural elements."""
        structural_seed = self.config.experiment_config.generate_structural_seed()
        return StructuralRNGManager(structural_seed)
    
    def _setup_operational_rng(self, replication_number=0):
        """Set up the random number generator for operational processes."""
        operational_base_seed = self.config.experiment_config.generate_operational_base_seed()
        return OperationalRNGManager(operational_base_seed, replication_number=replication_number)
    
    def _setup_repositories(self):
        """
        Initialize repositories for the current replication.
        
        This method preserves the restaurant repository (geographical layout) from
        structural initialization while creating fresh repositories for dynamic
        entities that change with each replication.
        """
        return {
            # Reuse existing restaurant repository to maintain constant
            # geographical layout across replications - this ensures that
            # differences between replications are due to operational
            # randomness, not structural differences
            'restaurant': self.restaurant_repository,
            
            # Create fresh repositories for dynamic entities that should
            # be generated anew for each replication
            'order': OrderRepository(),
            'driver': DriverRepository(),
            'pair': PairRepository(),
            'delivery_unit': DeliveryUnitRepository()
        }
    
    def _setup_restaurant_infrastructure(self):
        """
        Create and initialize restaurant infrastructure using uniform random distribution.
        """
        # Extract parameters from config
        area_size = self.config.structural_config.delivery_area_size
        num_restaurants = self.config.structural_config.num_restaurants
        
        # Generate random restaurant locations
        restaurants = []
        for i in range(num_restaurants):
            # Generate random coordinates within delivery area
            location = self.structural_rng.generate_uniform(0, area_size, size=2).tolist()
            
            # Create restaurant with unique ID and generated location
            restaurant = Restaurant(restaurant_id=i, location=location)
            
            # Add to repository and collection
            self.restaurant_repository.add(restaurant)
            restaurants.append(restaurant)
        
        print(f"Created {num_restaurants} restaurants with uniform random distribution")
        
        return restaurants
    
    def _setup_services(self):
        """Initialize and connect all services."""
        services = {}
        
        # Create order arrival service
        services['order_arrival'] = OrderArrivalService(
            env=self.env,
            event_dispatcher=self.event_dispatcher,
            order_repository=self.repositories['order'],
            restaurant_repository=self.repositories['restaurant'],
            config=self.flat_config,
            id_generator=self.id_generators['order'],
            operational_rng_manager=self.operational_rng
        )
        
        # Create driver arrival service
        services['driver_arrival'] = DriverArrivalService(
            env=self.env,
            event_dispatcher=self.event_dispatcher,
            driver_repository=self.repositories['driver'],
            config=self.flat_config,
            id_generator=self.id_generators['driver'],
            operational_rng_manager=self.operational_rng
        )
        
        # Create pairing service if enabled
        if self.config.operational_config.pairing_enabled:
            services['pairing'] = PairingService(
                env=self.env,
                event_dispatcher=self.event_dispatcher,
                order_repository=self.repositories['order'],
                pair_repository=self.repositories['pair'],
                config=self.flat_config
            )
        
        # Create assignment service
        services['assignment'] = AssignmentService(
            env=self.env,
            event_dispatcher=self.event_dispatcher,
            order_repository=self.repositories['order'],
            driver_repository=self.repositories['driver'],
            pair_repository=self.repositories['pair'],
            delivery_unit_repository=self.repositories['delivery_unit'],
            config=self.flat_config
        )
        
        # Create delivery service
        services['delivery'] = DeliveryService(
            env=self.env,
            event_dispatcher=self.event_dispatcher,
            driver_repository=self.repositories['driver'],
            order_repository=self.repositories['order'],
            pair_repository=self.repositories['pair'],
            delivery_unit_repository=self.repositories['delivery_unit'],
            config=self.flat_config
        )
        
        # Create driver scheduling service
        services['driver_scheduling'] = DriverSchedulingService(
            env=self.env,
            event_dispatcher=self.event_dispatcher,
            driver_repository=self.repositories['driver']
        )
        
        print(f"Initialized {len(services)} services")
        
        return services