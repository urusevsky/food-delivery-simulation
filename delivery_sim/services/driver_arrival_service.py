from delivery_sim.entities.driver import Driver
from delivery_sim.events.driver_events import DriverLoggedInEvent

class DriverArrivalService:
    """
    Service responsible for generating new drivers entering the system.
    
    This service runs as a continuous SimPy process, creating new drivers
    based on configured inter-arrival times and dispatching events when
    drivers log in.
    """
    
    def __init__(self, env, event_dispatcher, driver_repository, config, id_generator):
        """
        Initialize the driver arrival service.
        
        Args:
            env: SimPy environment
            event_dispatcher: Central event dispatcher
            driver_repository: Repository for storing created drivers
            config: Configuration containing arrival rate parameters
            id_generator: Generator for unique driver IDs
        """
        self.env = env
        self.event_dispatcher = event_dispatcher
        self.driver_repository = driver_repository
        self.config = config
        self.id_generator = id_generator
        
        # Start the arrival process
        self.process = env.process(self._arrival_process())
    
    def _arrival_process(self):
        """SimPy process that generates new drivers at configured intervals."""
        while True:
            # Generate time until next driver arrival
            inter_arrival_time = self._generate_inter_arrival_time()
            yield self.env.timeout(inter_arrival_time)
            
            # Generate driver attributes
            driver_id = self.id_generator.next()
            initial_location = self._generate_initial_location()
            service_duration = self._generate_service_duration()
            
            # Create new driver
            new_driver = Driver(
                driver_id=driver_id,
                initial_location=initial_location,
                login_time=self.env.now,
                service_duration=service_duration
            )
            
            # Add to repository
            self.driver_repository.add(new_driver)
            
            # Dispatch driver logged in event
            self.event_dispatcher.dispatch(DriverLoggedInEvent(
                timestamp=self.env.now,
                driver_id=driver_id,
                initial_location=initial_location,
                service_duration=service_duration
            ))
            
            # Log for debugging
            print(f"Driver {driver_id} logged in at time {self.env.now}")
    
    def _generate_inter_arrival_time(self):
        """
        Generate the time until the next driver arrival.
        
        In a complete implementation, this would use a random distribution.
        
        Returns:
            float: Time until next arrival in minutes
        """
        # Placeholder: In a real implementation, this would use a distribution
        return self.config.mean_driver_inter_arrival_time
    
    def _generate_initial_location(self):
        """
        Generate an initial location for a new driver.
        
        In a complete implementation, this would use a spatial distribution.
        
        Returns:
            list: [x, y] coordinates
        """
        # Placeholder: In a real implementation, this would use a distribution
        return [10, 10]
    
    def _generate_service_duration(self):
        """
        Generate a service duration for a new driver.
        
        In a complete implementation, this would use a distribution.
        
        Returns:
            float: Service duration in minutes
        """
        # Placeholder: In a real implementation, this would use a distribution
        return 120  # minutes