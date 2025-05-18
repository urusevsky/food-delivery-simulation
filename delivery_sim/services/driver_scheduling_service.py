from delivery_sim.entities.states import DriverState
from delivery_sim.events.driver_events import DriverLoggedInEvent, DriverLogoutAttemptEvent, DriverLoggedOutEvent
from delivery_sim.events.delivery_unit_events import DeliveryUnitCompletedEvent
from delivery_sim.utils.logging_system import get_logger

class DriverSchedulingService:
    def __init__(self, env, event_dispatcher, driver_repository):
        """Initialize the driver scheduling service with its dependencies."""
        # Get a logger instance specific to this component
        self.logger = get_logger("service.driver_scheduling")
        
        # Store dependencies
        self.env = env
        self.event_dispatcher = event_dispatcher
        self.driver_repository = driver_repository
        
        # Log service initialization
        self.logger.info(f"[t={self.env.now:.2f}] DriverSchedulingService initialized")
        
        # Register for relevant events
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Registering handler for DriverLoggedInEvent")
        self.event_dispatcher.register(DriverLoggedInEvent, self.handle_driver_login)
        
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Registering handler for DeliveryUnitCompletedEvent")
        self.event_dispatcher.register(DeliveryUnitCompletedEvent, self.handle_delivery_completed)
        
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Registering handler for DriverLogoutAttemptEvent")
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
        # Log event handling
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Handling DriverLoggedInEvent for driver {event.driver_id}")
        
        # Extract identifiers from event
        driver_id = event.driver_id
        intended_logout_time = event.timestamp + event.service_duration
        
        # Validate driver exists
        driver = self.driver_repository.find_by_id(driver_id)
        if not driver:
            self.logger.validation(f"[t={self.env.now:.2f}] Driver {driver_id} not found, cannot schedule logout")
            return
        
        # Pass validated information to operation
        self.schedule_driver_logout(driver_id, intended_logout_time)
        self.logger.debug(f"[t={self.env.now:.2f}] Scheduled logout for driver {driver_id} at time {intended_logout_time:.2f}")
    
    def handle_delivery_completed(self, event):
        """
        Handler for DeliveryUnitCompletedEvent. Validates driver and checks logout conditions.
        
        Args:
            event: The DeliveryUnitCompletedEvent
        """
        # Log event handling
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Handling DeliveryUnitCompletedEvent for unit {event.delivery_unit_id} by driver {event.driver_id}")
        
        # Extract identifiers from event
        driver_id = event.driver_id
        timestamp = event.timestamp
        
        # Validate driver exists
        driver = self.driver_repository.find_by_id(driver_id)
        if not driver:
            self.logger.validation(f"[t={self.env.now:.2f}] Driver {driver_id} not found, cannot check logout conditions")
            return
        
        # Pass validated entity to operation
        self.check_overdue_logout(driver, timestamp)
    
    def handle_driver_logout_attempt(self, event):
        """
        Handler for scheduled logout attempts at a driver's intended time.
        
        Args:
            event: The DriverLogoutAttemptEvent
        """
        # Log event handling
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Handling DriverLogoutAttemptEvent for driver {event.driver_id}")
        
        # Extract identifiers from event
        driver_id = event.driver_id
        timestamp = event.timestamp
        
        # Validate driver exists
        driver = self.driver_repository.find_by_id(driver_id)
        if not driver:
            self.logger.validation(f"[t={self.env.now:.2f}] Driver {driver_id} not found, cannot attempt logout")
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
        self.logger.info(f"[t={self.env.now:.2f}] Scheduling logout for driver {driver_id} at time {intended_logout_time:.2f}")
        self.env.process(self._driver_logout_process(driver_id, intended_logout_time))
    
    def check_overdue_logout(self, driver, current_time):
        """
        Check if a driver should log out after completing a delivery.
        
        This operation assumes the driver has been validated and focuses on business logic.
        
        Args:
            driver: The validated Driver object
            current_time: Current simulation time
        """
        # Business logic: Check if driver is past intended logout time
        if current_time >= driver.intended_logout_time:
            self.logger.info(f"[t={self.env.now:.2f}] Driver {driver.driver_id} has exceeded intended logout time, attempting immediate logout")
            
            # Dispatch logout attempt event
            self.logger.simulation_event(f"[t={self.env.now:.2f}] Dispatching DriverLogoutAttemptEvent for driver {driver.driver_id}")
            self.event_dispatcher.dispatch(DriverLogoutAttemptEvent(
                timestamp=current_time,
                driver_id=driver.driver_id
            ))
        else:
            self.logger.debug(f"[t={self.env.now:.2f}] Driver {driver.driver_id} has not reached intended logout time yet")
    
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
        # Business logic: Check if driver can log out
        if driver.can_logout():
            # Update driver state
            self.logger.info(f"[t={self.env.now:.2f}] Logging out driver {driver.driver_id}")
            driver.transition_to(DriverState.OFFLINE, self.event_dispatcher, self.env)
            
            # Dispatch completion event
            self.logger.simulation_event(f"[t={self.env.now:.2f}] Dispatching DriverLoggedOutEvent for driver {driver.driver_id}")
            self.event_dispatcher.dispatch(DriverLoggedOutEvent(
                timestamp=timestamp,
                driver_id=driver.driver_id,
                final_location=driver.location,
                login_time=driver.login_time
            ))
            
            return True
        else:
            self.logger.info(f"[t={self.env.now:.2f}] Cannot log out driver {driver.driver_id}: current state is {driver.state}")
            return False
    
    # SimPy process (internal method)
    def _driver_logout_process(self, driver_id, intended_logout_time):
        """SimPy process that schedules a logout attempt at the intended time."""
        self.logger.debug(f"[t={self.env.now:.2f}] Started driver logout process for driver {driver_id} with intended logout at {intended_logout_time:.2f}")
        
        time_until_logout = intended_logout_time - self.env.now
        if time_until_logout > 0:
            self.logger.debug(f"[t={self.env.now:.2f}] Waiting {time_until_logout:.2f} minutes until driver {driver_id}'s logout time")
            yield self.env.timeout(time_until_logout)
        
        self.logger.debug(f"[t={self.env.now:.2f}] Driver {driver_id}'s intended logout time reached")
        self.logger.simulation_event(f"[t={self.env.now:.2f}] Dispatching DriverLogoutAttemptEvent for driver {driver_id}")
        self.event_dispatcher.dispatch(DriverLogoutAttemptEvent(
            timestamp=self.env.now,
            driver_id=driver_id
        ))