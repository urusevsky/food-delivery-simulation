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
from delivery_sim.utils.id_generator import PrefixedIdGenerator
from delivery_sim.utils.logging_system import get_logger, configure_logging


class SimulationRunner:
    """
    Orchestrates the initialization and execution of the food delivery simulation.
    
    This class handles setting up all necessary components, connecting them
    appropriately, and running the simulation for the specified duration.
    """
    def __init__(self, simulation_config):
        # Get a logger instance specific to this component
        self.logger = get_logger("simulation.runner")
        
        # Store the hierarchical configuration
        self.config = simulation_config
        
        # Create a flat config wrapper for services
        self.flat_config = simulation_config.flat_config  # Easy access
        
        # Configure logging based on config
        configure_logging(simulation_config.logging_config)
        
        # Log initialization
        self.logger.info("SimulationRunner initialized")
        
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
        self.logger.info("Beginning simulation initialization")
        
        # First initialize structural components
        self.initialize_structural_components()
        
        # Then initialize operational components
        self.initialize_operational_components()
        
        self.logger.info("Simulation initialization complete")
        return self
    
    def initialize_structural_components(self):
        """Initialize components that remain constant across replications."""
        self.logger.info("Initializing structural components")
        
        # Initialize ID generators
        self.id_generators = self._setup_id_generators()
        
        # Initialize structural random number generator
        structural_seed = self.config.experiment_config.generate_structural_seed()
        self.structural_rng = StructuralRNGManager(structural_seed)
        self.logger.debug(f"Using structural seed: {structural_seed}")
        
        # Initialize restaurant repository
        self.restaurant_repository = RestaurantRepository()
        
        # Generate restaurant infrastructure
        self._setup_restaurant_infrastructure()
        
        self.logger.info("Structural components initialization complete")
        return self
    
    def initialize_operational_components(self, replication_number=0):
        """Initialize components that change with each replication."""
        self.logger.info(f"Initializing operational components for replication {replication_number}")
        
        # Create SimPy environment and event dispatcher
        self.env = simpy.Environment()
        self.event_dispatcher = EventDispatcher()
        
        # Set up operational random number generator
        operational_base_seed = self.config.experiment_config.generate_operational_base_seed()
        self.operational_rng = OperationalRNGManager(operational_base_seed, replication_number)
        self.logger.debug(f"Using operational base seed: {operational_base_seed} for replication {replication_number}")
        
        # Initialize repositories
        self.repositories = self._setup_repositories()
        
        # Initialize services
        self.services = self._setup_services()
        
        self.logger.info(f"Operational components initialization complete for replication {replication_number}")
        return self
    
    def run(self):
        """Run the simulation for the configured duration."""
        duration = self.config.experiment_config.simulation_duration
        self.logger.info(f"Starting simulation run for {duration} minutes")
        
        # Run the simulation
        self.env.run(until=duration)
        
        # Log final simulation time to mark completion
        self.logger.info(f"Simulation completed after {self.env.now:.2f} minutes")
        
        # Log summary statistics
        self._log_simulation_summary()
        
        return self.repositories
    
    def _log_simulation_summary(self):
        """Log summary statistics of the simulation run."""
        # Order statistics
        orders = self.order_repository.find_all()
        completed_orders = len([o for o in orders if o.state == 'delivered'])
        total_orders = len(orders)
        
        # Driver statistics
        drivers = self.driver_repository.find_all()
        total_drivers = len(drivers)
        
        # Pair statistics if pairing enabled
        if self.config.operational_config.pairing_enabled:
            pairs = self.pair_repository.find_all()
            completed_pairs = len([p for p in pairs if p.state == 'completed'])
            total_pairs = len(pairs)
            pair_ratio = len(pairs) / len(orders) if orders else 0
            
            self.logger.info(f"Simulation summary: "
                          f"{completed_orders}/{total_orders} orders completed, "
                          f"{completed_pairs}/{total_pairs} pairs completed, "
                          f"pair ratio: {pair_ratio:.2f}, "
                          f"total drivers: {total_drivers}")
        else:
            self.logger.info(f"Simulation summary: "
                          f"{completed_orders}/{total_orders} orders completed, "
                          f"total drivers: {total_drivers}")
    
    def _setup_id_generators(self):
        """
        Set up ID generators for each entity type.
        
        We use a consistent prefixing scheme:
        - O for Orders (O1, O2, ...)
        - D for Drivers (D1, D2, ...)  
        - R for Restaurants (R1, R2, ...)
        
        This makes IDs immediately recognizable while keeping them concise.
        """

        id_generators = {
            'order': PrefixedIdGenerator('O', 1),      # Generates: O1, O2, O3, ...
            'driver': PrefixedIdGenerator('D', 1),     # Generates: D1, D2, D3, ...
            'restaurant': PrefixedIdGenerator('R', 1)  # Generates: R1, R2, R3, ...
        }
        
        self.logger.debug(f"Created ID generators for: {', '.join(id_generators.keys())}")
        self.logger.debug("ID format - Orders: O#, Drivers: D#, Restaurants: R#")
        
        return id_generators
    
    def _setup_repositories(self):
        """
        Initialize repositories for the current replication.
        
        This method preserves the restaurant repository (geographical layout) from
        structural initialization while creating fresh repositories for dynamic
        entities that change with each replication.
        """
        # Store repositories for convenient access (used in _log_simulation_summary)
        repositories = {
            # Reuse existing restaurant repository
            'restaurant': self.restaurant_repository,
            
            # Create fresh repositories for dynamic entities
            'order': OrderRepository(),
            'driver': DriverRepository(),
            'pair': PairRepository(),
            'delivery_unit': DeliveryUnitRepository()
        }
        
        # Store direct references for easier access in summary logging
        self.order_repository = repositories['order']
        self.driver_repository = repositories['driver']
        self.pair_repository = repositories['pair']
        
        self.logger.debug(f"Initialized repositories: {', '.join(repositories.keys())}")
        return repositories
    
    def _setup_restaurant_infrastructure(self):
        """
        Create and initialize restaurant infrastructure using uniform random distribution.
        
        Restaurants are assigned IDs using the prefixed format (R1, R2, etc.)
        for consistency with other entities in the system.
        """
        # Extract parameters from config
        area_size = self.config.structural_config.delivery_area_size
        num_restaurants = self.config.structural_config.num_restaurants
        
        self.logger.info(f"Creating {num_restaurants} restaurants in delivery area of {area_size}x{area_size} km")
        
        # Generate random restaurant locations
        restaurants = []
        for i in range(num_restaurants):
            # Generate random coordinates within delivery area
            location = self.structural_rng.generate_uniform(0, area_size, size=2).tolist()
            
            # Get the next restaurant ID from our generator
            restaurant_id = self.id_generators['restaurant'].next()
            
            # Create restaurant with generated ID and location
            restaurant = Restaurant(restaurant_id=restaurant_id, location=location)
            
            # Add to repository and collection
            self.restaurant_repository.add(restaurant)
            restaurants.append(restaurant)
        
        self.logger.info(f"Created {num_restaurants} restaurants with IDs {restaurants[0].restaurant_id} through {restaurants[-1].restaurant_id}")
        return restaurants
    
    def _setup_services(self):
        """Initialize and connect all services."""
        # Create core services
        services = {
            'order_arrival': OrderArrivalService(
                env=self.env,
                event_dispatcher=self.event_dispatcher,
                order_repository=self.repositories['order'],
                restaurant_repository=self.repositories['restaurant'],
                config=self.flat_config,
                id_generator=self.id_generators['order'],
                operational_rng_manager=self.operational_rng
            ),
            
            'driver_arrival': DriverArrivalService(
                env=self.env,
                event_dispatcher=self.event_dispatcher,
                driver_repository=self.repositories['driver'],
                config=self.flat_config,
                id_generator=self.id_generators['driver'],
                operational_rng_manager=self.operational_rng
            ),
            
            'assignment': AssignmentService(
                env=self.env,
                event_dispatcher=self.event_dispatcher,
                order_repository=self.repositories['order'],
                driver_repository=self.repositories['driver'],
                pair_repository=self.repositories['pair'],
                delivery_unit_repository=self.repositories['delivery_unit'],
                config=self.flat_config
            ),
            
            'delivery': DeliveryService(
                env=self.env,
                event_dispatcher=self.event_dispatcher,
                driver_repository=self.repositories['driver'],
                order_repository=self.repositories['order'],
                pair_repository=self.repositories['pair'],
                delivery_unit_repository=self.repositories['delivery_unit'],
                config=self.flat_config
            ),
            
            'driver_scheduling': DriverSchedulingService(
                env=self.env,
                event_dispatcher=self.event_dispatcher,
                driver_repository=self.repositories['driver']
            )
        }
        
        # Create pairing service if enabled
        if self.config.operational_config.pairing_enabled:
            services['pairing'] = PairingService(
                env=self.env,
                event_dispatcher=self.event_dispatcher,
                order_repository=self.repositories['order'],
                pair_repository=self.repositories['pair'],
                config=self.flat_config
            )
        
        # Log services created
        pairing_status = "with pairing" if self.config.operational_config.pairing_enabled else "without pairing"
        self.logger.info(f"Initialized {len(services)} services {pairing_status}")
        
        return services