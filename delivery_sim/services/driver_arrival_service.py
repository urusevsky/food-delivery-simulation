from delivery_sim.entities.driver import Driver
from delivery_sim.events.driver_events import DriverLoggedInEvent
from delivery_sim.utils.logging_system import get_logger
import numpy as np
from scipy.stats import lognorm

class DriverArrivalService:
    """
    Service responsible for generating new drivers entering the system.
    
    This service runs as a continuous SimPy process, creating new drivers
    based on configured inter-arrival times and dispatching events when
    drivers log in.
    """
    
    def __init__(self, env, event_dispatcher, driver_repository, config, id_generator, operational_rng_manager):
        """
        Initialize the driver arrival service.
        
        Args:
            env: SimPy environment
            event_dispatcher: Central event dispatcher
            driver_repository: Repository for storing created drivers
            config: Configuration containing arrival rate parameters
            id_generator: Generator for unique driver IDs
            operational_rng_manager: Manager for random number streams
        """
        # Get a logger instance specific to this component
        self.logger = get_logger("services.driver_arrival")
        
        # Store dependencies
        self.env = env
        self.event_dispatcher = event_dispatcher
        self.driver_repository = driver_repository
        self.config = config
        self.id_generator = id_generator
        
        # Get all random streams at initialization time
        self.arrival_stream = operational_rng_manager.get_stream('driver_arrivals')
        self.location_stream = operational_rng_manager.get_stream('driver_initial_locations')
        self.service_duration_stream = operational_rng_manager.get_stream('service_duration')
        
        # Log service initialization with configuration details
        self.logger.info(f"[t={self.env.now:.2f}] DriverArrivalService initialized with mean inter-arrival time: {config.mean_driver_inter_arrival_time} minutes")
        
        # Start the arrival process
        self.logger.info(f"[t={self.env.now:.2f}] Starting driver arrival process")
        self.process = env.process(self._arrival_process())
    
    def _arrival_process(self):
        """SimPy process that generates new drivers at configured intervals."""
        while True:
            # Generate time until next driver arrival
            inter_arrival_time = self._generate_inter_arrival_time()
            self.logger.debug(f"[t={self.env.now:.2f}] Next driver will arrive in {inter_arrival_time:.2f} minutes")
            
            yield self.env.timeout(inter_arrival_time)
            
            # Generate driver attributes
            driver_id = self.id_generator.next()
            initial_location = self._generate_initial_location()
            service_duration = self._generate_service_duration()
            
            self.logger.debug(f"[t={self.env.now:.2f}] Generated attributes for driver {driver_id}: "
                            f"location={initial_location}, service_duration={service_duration:.2f}")
            
            # Create new driver
            new_driver = Driver(
                driver_id=driver_id,
                initial_location=initial_location,
                login_time=self.env.now,
                service_duration=service_duration
            )
            
            # Add to repository
            self.driver_repository.add(new_driver)
            
            # Log driver creation
            self.logger.info(f"[t={self.env.now:.2f}] Created driver {driver_id} at location {initial_location} with service duration {service_duration:.2f} minutes")
            
            # Dispatch driver logged in event
            self.logger.simulation_event(f"[t={self.env.now:.2f}] Dispatching DriverLoggedInEvent for driver {driver_id}")
            self.event_dispatcher.dispatch(DriverLoggedInEvent(
                timestamp=self.env.now,
                driver_id=driver_id,
                initial_location=initial_location,
                service_duration=service_duration
            ))
    
    def _generate_inter_arrival_time(self):
        """
        Generate the time until the next driver arrival using an exponential distribution.
        
        This models arrivals as a Poisson process, which is standard for independent
        arrivals in service systems.
        
        Returns:
            float: Time until next arrival in minutes
        """
        return self.arrival_stream.exponential(self.config.mean_driver_inter_arrival_time)
    
    def _generate_initial_location(self):
        """
        Generate an initial location for a new driver.
        
        This uses a uniform distribution across the delivery area.
        In a more sophisticated model, this might use hotspots or other spatial distributions.
        
        Returns:
            list: [x, y] coordinates
        """
        area_size = self.config.delivery_area_size
        return self.location_stream.uniform(0, area_size, size=2).tolist()
    
    def _generate_service_duration(self):
        """
        Generate a service duration for a new driver using a truncated lognormal distribution.
        
        Lognormal is a common distribution for service times as it:
        1. Is always positive
        2. Has a long right tail (some drivers work much longer than average)
        3. Has more mass near the mean than exponential
        
        We truncate it to ensure values stay within reasonable bounds.
        
        Returns:
            float: Service duration in minutes
        """
        # Extract parameters from config
        mean = self.config.mean_service_duration
        std_dev = self.config.service_duration_std_dev
        min_duration = self.config.min_service_duration
        max_duration = self.config.max_service_duration
        
        # Calculate lognormal parameters
        sigma_squared = np.log(1 + (std_dev / mean) ** 2)
        sigma = np.sqrt(sigma_squared)
        mu = np.log(mean) - sigma_squared / 2
        
        # Create distribution and calculate bounds
        distribution = lognorm(s=sigma, scale=np.exp(mu))
        cdf_lower = distribution.cdf(min_duration)
        cdf_upper = distribution.cdf(max_duration)
        
        # Generate truncated lognormal using inverse CDF method
        service_duration = distribution.ppf(self.service_duration_stream.uniform(cdf_lower, cdf_upper))
        
        self.logger.debug(f"[t={self.env.now:.2f}] Generated service duration {service_duration:.2f} (min={min_duration}, max={max_duration})")
        return service_duration