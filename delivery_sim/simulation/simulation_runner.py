# delivery_sim/simulation/simulation_runner.py
"""
SimulationRunner: Clean and simple runner for single configuration, multi-replication experiments.

This runner separates invariant components (same across replications) from variant components 
(fresh per replication) for efficiency and experimental control.
"""

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
from delivery_sim.utils.infrastructure_analysis import analyze_infrastructure
from delivery_sim.utils.priority_scoring import create_priority_scorer
from delivery_sim.system_data.system_data_definitions import SystemDataDefinitions
from delivery_sim.system_data.system_data_collector import SystemDataCollector
from delivery_sim.system_data.system_snapshot_repository import SystemSnapshotRepository

class SimulationRunner:
    """
    Clean and simple runner for multi-replication experiments.
    
    Efficiently handles multiple replications by separating:
    - Invariant components: Same across replications (experimental control)
    - Variant components: Fresh per replication (statistical independence)
    """
    
    def __init__(self):
        self.logger = get_logger("simulation.runner")
        
        # These will be set during initialization
        self.config = None
        
        # Invariant components (same across replications)
        self.restaurant_repository = None
        self.infrastructure_characteristics = None
        self.structural_rng = None
        self.priority_scorer = None
        self.id_generators = None
        
        # Variant components (fresh per replication)
        self.env = None
        self.event_dispatcher = None
        self.order_repository = None
        self.driver_repository = None
        self.pair_repository = None
        self.delivery_unit_repository = None
        self.operational_rng = None
        self.services = None

        # NEW: System data collection components (variant per replication)
        self.system_data_definitions = None
        self.system_data_collector = None
        self.system_snapshot_repository = None
    
    def run_experiment(self, config):
        """
        Run complete experiment for a single configuration with multiple replications.
        
        Args:
            config: Complete simulation configuration
            
        Returns:
            dict: Experiment results with replication data and infrastructure info
        """
        self.config = config
        
        # Configure logging first
        configure_logging(config.logging_config)
        
        self.logger.info(f"Starting experiment with {config.experiment_config.num_replications} replications")
        self.logger.debug(f"Full configuration:\n{config}")
        
        # ===== PHASE 1: INITIALIZE INVARIANT COMPONENTS =====
        self.logger.info("Phase 1: Initializing invariant components...")
        self._initialize_invariant_components()
        
        # ===== PHASE 2: RUN MULTIPLE REPLICATIONS =====
        self.logger.info(f"Phase 2: Running {config.experiment_config.num_replications} replications...")
        replication_results = []
        
        for replication in range(config.experiment_config.num_replications):
            self.logger.info(f"Starting replication {replication + 1}/{config.experiment_config.num_replications}")
            
            # Initialize fresh components for this replication
            self._initialize_variant_components(replication)
            
            # Run this replication
            replication_result = self._run_single_replication()
            replication_results.append(replication_result)
            
            self.logger.info(f"Completed replication {replication + 1}")
        
        # ===== PHASE 3: RETURN RESULTS =====
        experiment_result = {
            'replication_results': replication_results,  # Now includes system_snapshots
            'infrastructure_characteristics': self.infrastructure_characteristics,
            'config_summary': str(config),
            'num_replications': len(replication_results)
        }
        
        self.logger.info(f"Experiment completed: {len(replication_results)} replications")
        return experiment_result
    
    def _initialize_invariant_components(self):
        """
        Initialize components that remain the same across all replications.
        
        This ensures experimental control and consistency.
        """
        self.logger.debug("Initializing invariant components...")
        
        # 1. Create structural components (restaurants, structural RNG)
        self._create_structural_components()
        
        # 2. Analyze infrastructure (expensive Monte Carlo sampling - done once!)
        self._analyze_infrastructure()
        
        # 3. Create reusable priority scorer
        self._create_priority_scorer()
        
        self.logger.info("Invariant components initialized successfully")
    
    def _create_structural_components(self):
        """Create structural components (restaurants, RNG)."""
        self.logger.debug("Creating structural components...")
        
        # Create structural RNG for deterministic infrastructure
        structural_seed = self.config.experiment_config.master_seed
        self.structural_rng = StructuralRNGManager(structural_seed)
        self.logger.debug(f"Using structural seed: {structural_seed}")
        
        # Generate restaurants deterministically
        self.restaurant_repository = RestaurantRepository()
        restaurants = self._generate_restaurants(
            count=self.config.structural_config.num_restaurants,
            area_size=self.config.structural_config.delivery_area_size,
            rng=self.structural_rng.rng
        )
        
        for restaurant in restaurants:
            self.restaurant_repository.add(restaurant)
        
        self.logger.debug(f"Created {len(restaurants)} restaurants in "
                         f"{self.config.structural_config.delivery_area_size}x{self.config.structural_config.delivery_area_size}km area")
    
    def _analyze_infrastructure(self):
        """Perform infrastructure analysis (expensive Monte Carlo sampling)."""
        self.logger.debug("Analyzing infrastructure characteristics...")
        
        # Get scoring config if available for analysis parameters
        scoring_config = getattr(self.config, 'scoring_config', None)
        
        # Perform comprehensive infrastructure analysis
        self.infrastructure_characteristics = analyze_infrastructure(
            restaurant_repository=self.restaurant_repository,
            structural_config=self.config.structural_config,
            structural_rng=self.structural_rng,
            scoring_config=scoring_config
        )
        
        self.logger.info(f"Infrastructure analysis complete: {self.infrastructure_characteristics}")
    
    def _create_priority_scorer(self):
        """Create reusable priority scorer."""
        self.logger.debug("Priority scorer will be created per replication (variant component)")
        
        # Priority scorer is now variant - no creation here
        # Just validate that scoring config is available if needed
        if hasattr(self.config, 'scoring_config') and self.config.scoring_config:
            self.logger.debug("Scoring configuration available - priority scorer will be created per replication")
        else:
            self.logger.debug("No scoring configuration provided - priority scorer disabled")

    def _create_id_generators(self):
        """Create ID generators for entities that reset per replication."""
        self.logger.debug("Creating ID generators...")
        
        self.id_generators = {
            'order': PrefixedIdGenerator('O', 1),      # Generates: O1, O2, O3, ...
            'driver': PrefixedIdGenerator('D', 1),     # Generates: D1, D2, D3, ...
            # Restaurant IDs are handled separately in structural components
        }
    
    def _initialize_variant_components(self, replication_number):
        """
        Initialize fresh components for a single replication.
        
        This is fast since invariant components are reused.
        """
        self.logger.debug(f"Initializing variant components for replication {replication_number}...")
        
        # 1. Create fresh ID generators for this replication
        self._create_id_generators()
        
        # 2. Create fresh simulation environment
        self.env = simpy.Environment()
        self.event_dispatcher = EventDispatcher()
        
        # 3. Create fresh operational RNG (different seed per replication)
        operational_base_seed = self.config.experiment_config.master_seed
        self.operational_rng = OperationalRNGManager(operational_base_seed, replication_number)

        # Log sample seeds to verify replication independence
        sample_seeds = self.operational_rng.get_sample_stream_seeds()
        self.logger.debug(f"Replication {replication_number} RNG streams: {sample_seeds}")
        
        # 4. Create fresh entity repositories
        self.order_repository = OrderRepository()
        self.driver_repository = DriverRepository()
        self.pair_repository = PairRepository()
        self.delivery_unit_repository = DeliveryUnitRepository()
        
        # 5. Create priority scorer for this replication
        self.priority_scorer = create_priority_scorer(
                infrastructure_characteristics=self.infrastructure_characteristics,
                scoring_config=self.config.scoring_config,
                env=self.env
            )
        self.logger.debug(f"Priority scorer created for replication {replication_number}")
        
        # 6. Create services (connect invariant logic to variant environment)
        self._create_services()

        # NEW: Initialize system data collection infrastructure
        self._initialize_system_data_collection()
        
        self.logger.debug(f"Variant components initialized for replication {replication_number}")
    
    def _create_services(self):
        """Create services that connect invariant logic to variant environment."""
        self.logger.debug("Creating services...")
        
        # Create core services
        services = {
            'order_arrival': OrderArrivalService(
                env=self.env,
                event_dispatcher=self.event_dispatcher,
                order_repository=self.order_repository,
                restaurant_repository=self.restaurant_repository,  # Reuse invariant restaurants!
                config=self.config.flat_config,
                id_generator=self.id_generators['order'],  # Reuse invariant ID generator!
                operational_rng_manager=self.operational_rng
            ),
            
            'driver_arrival': DriverArrivalService(
                env=self.env,
                event_dispatcher=self.event_dispatcher,
                driver_repository=self.driver_repository,
                config=self.config.flat_config,
                id_generator=self.id_generators['driver'],  # Reuse invariant ID generator!
                operational_rng_manager=self.operational_rng
            ),

            'driver_scheduling': DriverSchedulingService(
                env=self.env,
                event_dispatcher=self.event_dispatcher,
                driver_repository=self.driver_repository
            ),

            'assignment': AssignmentService(
                env=self.env,
                event_dispatcher=self.event_dispatcher,
                order_repository=self.order_repository,
                driver_repository=self.driver_repository,
                pair_repository=self.pair_repository,
                delivery_unit_repository=self.delivery_unit_repository,
                priority_scorer=self.priority_scorer,  # Pass the priority scorer!
                config=self.config.flat_config
            ),
            
            'delivery': DeliveryService(
                env=self.env,
                event_dispatcher=self.event_dispatcher,
                driver_repository=self.driver_repository,
                order_repository=self.order_repository,
                pair_repository=self.pair_repository,
                delivery_unit_repository=self.delivery_unit_repository,
                restaurant_repository=self.restaurant_repository,  # Add this line
                config=self.config.flat_config
            )
        }
        
        # Create pairing service if enabled
        if self.config.operational_config.pairing_enabled:
            services['pairing'] = PairingService(
                env=self.env,
                event_dispatcher=self.event_dispatcher,
                order_repository=self.order_repository,
                pair_repository=self.pair_repository,
                config=self.config.flat_config
            )
        
        # Store services for easy access
        self.services = services
        
        # Log services created
        pairing_status = "with pairing" if self.config.operational_config.pairing_enabled else "without pairing"
        self.logger.info(f"Initialized {len(services)} services {pairing_status}")

    def _initialize_system_data_collection(self):
        """
        Initialize system data collection for warmup analysis.
        
        Creates the infrastructure needed to collect time series data
        during simulation for post-processing warmup detection.
        """
        self.logger.debug("Initializing system data collection infrastructure...")
        
        # Create repository for storing snapshots
        self.system_snapshot_repository = SystemSnapshotRepository()
        
        # Create definitions for calculating system metrics
        repositories_dict = {
            'order': self.order_repository,
            'driver': self.driver_repository,
            'pair': self.pair_repository,
            'delivery_unit': self.delivery_unit_repository
        }
        self.system_data_definitions = SystemDataDefinitions(repositories_dict)
        
        # Create collector that will run during simulation
        collection_interval = 0.5  # Collect every 0.5 simulation minutes for warmup analysis
        self.system_data_collector = SystemDataCollector(
            env=self.env,
            system_data_definitions=self.system_data_definitions,
            snapshot_repository=self.system_snapshot_repository,
            collection_interval=collection_interval
        )
        
        self.logger.debug(f"System data collection initialized with interval {collection_interval} minutes")

    def _run_single_replication(self):
        """
        Execute single replication and return raw repository data AND system snapshots.
        
        Enhanced to include system data collection results for warmup analysis.
        """
        # Run simulation (system data collector runs automatically as SimPy process)
        duration = self.config.experiment_config.simulation_duration
        self.logger.debug(f"Running simulation for {duration} minutes")
        self.env.run(until=duration)
        self.logger.info(f"Simulation completed at time {self.env.now:.2f}")
        
        # Collect repositories (existing functionality)
        repositories = {
            'order': self.order_repository,
            'driver': self.driver_repository,
            'pair': self.pair_repository,
            'delivery_unit': self.delivery_unit_repository,
            'restaurant': self.restaurant_repository  
        }

        # NEW: Collect system snapshots for warmup analysis
        system_snapshots = self.system_snapshot_repository.get_all_snapshots()
        
        self.logger.debug(f"Collected {len(system_snapshots)} system snapshots for warmup analysis")
        
        # Return enhanced results structure
        return {
            'repositories': repositories,
            'system_snapshots': system_snapshots  # NEW: Include time series data
        }
      
    def _generate_restaurants(self, count, area_size, rng):
        """Generate restaurant locations using structural RNG."""
        from delivery_sim.utils.location_utils import format_location
        
        restaurants = []
        for i in range(count):
            location = rng.uniform(0, area_size, size=2).tolist()
            restaurant_id = f"R{i+1}"
            restaurant = Restaurant(restaurant_id=restaurant_id, location=location)
            restaurants.append(restaurant)
            
            # Log restaurant creation
            self.logger.debug(f"Created restaurant {restaurant_id} at {format_location(location)}")
        
        self.logger.info(f"Generated {count} restaurants in {area_size}x{area_size}km area")
        return restaurants