from delivery_sim.entities.states import DriverState
from delivery_sim.events.driver_events import DriverLoggedInEvent, DriverStateChangedEvent, DriverLogoutAttemptEvent, DriverLoggedOutEvent

class DriverSchedulingService:
    def __init__(self, env, event_dispatcher, driver_repository):
        self.env = env
        self.event_dispatcher = event_dispatcher
        self.driver_repository = driver_repository
        
        # Register for relevant events
        self.event_dispatcher.register(DriverLoggedInEvent, self.handle_driver_login)
        self.event_dispatcher.register(DriverStateChangedEvent, self.handle_driver_state_changed)
        self.event_dispatcher.register(DriverLogoutAttemptEvent, self.handle_driver_logout_attempt)
    
    # Event Handlers
    def handle_driver_login(self, event):
        """Handler for DriverLoggedInEvent"""
        driver_id = event.driver_id
        intended_logout_time = event.timestamp + event.service_duration
        
        self.schedule_driver_logout(driver_id, intended_logout_time)
    
    def handle_driver_state_changed(self, event):
        """Handler for DriverStateChangedEvent"""
        if event.new_state != DriverState.AVAILABLE:
            return
            
        driver_id = event.driver_id
        timestamp = event.timestamp
        
        self.check_overdue_logout(driver_id, timestamp)
    
    def handle_driver_logout_attempt(self, event):
        """Handler for DriverLogoutAttemptEvent"""
        driver_id = event.driver_id
        timestamp = event.timestamp
        
        self.attempt_driver_logout(driver_id, timestamp)
    
    # Operations
    def schedule_driver_logout(self, driver_id, intended_logout_time):
        """Operation to schedule a driver's logout monitoring process."""
        # Start a SimPy process to monitor this driver
        self.env.process(self._driver_logout_process(driver_id, intended_logout_time))
    
    def check_overdue_logout(self, driver_id, current_time):
        """Operation to check if a driver should log out after state change."""
        driver = self.driver_repository.find_by_id(driver_id)
        if not driver:
            return
            
        if current_time >= driver.intended_logout_time:
            self.event_dispatcher.dispatch(DriverLogoutAttemptEvent(
                timestamp=current_time,
                driver_id=driver_id
            ))
    
    def attempt_driver_logout(self, driver_id, timestamp):
        """Operation to attempt logging out a driver."""
        driver = self.driver_repository.find_by_id(driver_id)
        if not driver:
            return False
        
        if driver.can_logout():
            # Update driver state
            driver.transition_to(DriverState.OFFLINE, self.event_dispatcher, self.env)
            driver.actual_log_out_time = timestamp
            
            # Dispatch completion event
            self.event_dispatcher.dispatch(DriverLoggedOutEvent(
                timestamp=timestamp,
                driver_id=driver_id,
                final_location=driver.location,
                login_time=driver.login_time
            ))
            
            return True
        
        return False
    
    # SimPy process (internal method)
    def _driver_logout_process(self, driver_id, intended_logout_time):
        """SimPy process that monitors a driver's logout schedule."""
        time_until_logout = intended_logout_time - self.env.now
        if time_until_logout > 0:
            yield self.env.timeout(time_until_logout)
        
        self.event_dispatcher.dispatch(DriverLogoutAttemptEvent(
            timestamp=self.env.now,
            driver_id=driver_id
        ))