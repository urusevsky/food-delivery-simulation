from delivery_sim.entities.states import DriverState
from delivery_sim.events.driver_events import DriverLoggedInEvent, DriverLogoutAttemptEvent, DriverLoggedOutEvent
from delivery_sim.events.delivery_unit_events import DeliveryUnitCompletedEvent
from delivery_sim.utils.validation_utils import log_entity_not_found

class DriverSchedulingService:
    def __init__(self, env, event_dispatcher, driver_repository):
        """Initialize the driver scheduling service with its dependencies."""
        self.env = env
        self.event_dispatcher = event_dispatcher
        self.driver_repository = driver_repository
        
        # Register for relevant events
        self.event_dispatcher.register(DriverLoggedInEvent, self.handle_driver_login)
        self.event_dispatcher.register(DeliveryUnitCompletedEvent, self.handle_delivery_completed)
        self.event_dispatcher.register(DriverLogoutAttemptEvent, self.handle_driver_logout_attempt)
    
    # ===== Event Handlers (Entry Points) =====
    
    def handle_driver_login(self, event):
        """
        Handler for DriverLoggedInEvent. Validates driver and schedules logout if valid.
        
        This handler implements the Entry-Point Validation Pattern by validating
        entities before passing them to operations.
        
        Args:
            event: The DriverLoggedInEvent
        """
        # Extract identifiers from event
        driver_id = event.driver_id
        intended_logout_time = event.timestamp + event.service_duration
        
        # Validate driver exists
        driver = self.driver_repository.find_by_id(driver_id)
        if not driver:
            log_entity_not_found(self.__class__.__name__, "Driver", driver_id, self.env.now)
            return
        
        # Pass validated information to operation
        self.schedule_driver_logout(driver_id, intended_logout_time)
    
    def handle_delivery_completed(self, event):
        """
        Handler for DeliveryUnitCompletedEvent. Validates driver and checks logout conditions.
        
        Args:
            event: The DeliveryUnitCompletedEvent
        """
        # Extract identifiers from event
        driver_id = event.driver_id
        timestamp = event.timestamp
        
        # Validate driver exists
        driver = self.driver_repository.find_by_id(driver_id)
        if not driver:
            log_entity_not_found(self.__class__.__name__, "Driver", driver_id, self.env.now)
            return
        
        # Pass validated entity to operation
        self.check_overdue_logout(driver, timestamp)
    
    def handle_driver_logout_attempt(self, event):
        """
        Handler for scheduled logout attempts at a driver's intended time.
        
        Args:
            event: The DriverLogoutAttemptEvent
        """
        # Extract identifiers from event
        driver_id = event.driver_id
        timestamp = event.timestamp
        
        # Validate driver exists
        driver = self.driver_repository.find_by_id(driver_id)
        if not driver:
            log_entity_not_found(self.__class__.__name__, "Driver", driver_id, self.env.now)
            return
        
        # Pass validated entity to operation
        self.attempt_driver_logout(driver, timestamp)
    
    # ===== Operations (Business Logic) =====
    
    def schedule_driver_logout(self, driver_id, intended_logout_time):
        """
        Schedule a logout attempt for when the driver's intended time arrives.
        
        Args:
            driver_id: ID of the driver to schedule logout for
            intended_logout_time: When the driver intends to log out
        """
        self.env.process(self._driver_logout_process(driver_id, intended_logout_time))
    
    def check_overdue_logout(self, driver, current_time):
        """
        Check if a driver should log out after completing a delivery.
        
        This operation assumes the driver has been validated and focuses on business logic.
        
        Args:
            driver: The validated Driver object
            current_time: Current simulation time
        """
        if current_time >= driver.intended_logout_time:
            self.event_dispatcher.dispatch(DriverLogoutAttemptEvent(
                timestamp=current_time,
                driver_id=driver.driver_id
            ))
    
    def attempt_driver_logout(self, driver, timestamp):
        """
        Attempt to log out a driver if they're available.
        
        This operation assumes the driver has been validated and focuses on business logic.
        
        Args:
            driver: The validated Driver object
            timestamp: Current simulation time
            
        Returns:
            bool: True if driver logged out successfully, False otherwise
        """
        if driver.can_logout():
            # Update driver state
            driver.transition_to(DriverState.OFFLINE, self.event_dispatcher, self.env)
            
            # Dispatch completion event
            self.event_dispatcher.dispatch(DriverLoggedOutEvent(
                timestamp=timestamp,
                driver_id=driver.driver_id,
                final_location=driver.location,
                login_time=driver.login_time
            ))
            
            return True
        
        return False
    
    # SimPy process (internal method)
    def _driver_logout_process(self, driver_id, intended_logout_time):
        """SimPy process that schedules a logout attempt at the intended time."""
        time_until_logout = intended_logout_time - self.env.now
        if time_until_logout > 0:
            yield self.env.timeout(time_until_logout)
        
        self.event_dispatcher.dispatch(DriverLogoutAttemptEvent(
            timestamp=self.env.now,
            driver_id=driver_id
        ))