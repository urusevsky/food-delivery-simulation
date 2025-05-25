# tests/integration/test_driver_lifecycle_scheduling.py
import pytest
import simpy
from unittest.mock import patch, Mock

# Import core components
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.events.driver_events import (
    DriverLoggedInEvent, DriverLoggedOutEvent, DriverAvailableForAssignmentEvent
)
from delivery_sim.events.delivery_unit_events import DeliveryUnitCompletedEvent
from delivery_sim.repositories.driver_repository import DriverRepository
from delivery_sim.repositories.order_repository import OrderRepository
from delivery_sim.repositories.pair_repository import PairRepository
from delivery_sim.repositories.delivery_unit_repository import DeliveryUnitRepository
from delivery_sim.services.driver_scheduling_service import DriverSchedulingService
from delivery_sim.services.assignment_service import AssignmentService
from delivery_sim.entities.driver import Driver
from delivery_sim.entities.order import Order
from delivery_sim.entities.delivery_unit import DeliveryUnit
from delivery_sim.entities.states import DriverState, OrderState
from delivery_sim.utils.entity_type_utils import EntityType


class TestDriverSchedulingIntegration:
    """
    Integration tests for DriverSchedulingService interactions with other services.
    
    These tests verify:
    1. Driver login triggers logout scheduling
    2. Delivery completion triggers appropriate availability evaluation
    3. DriverAvailableForAssignmentEvent flows correctly to AssignmentService
    4. Scheduled logout behavior under different driver states
    """
    
    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        class TestConfig:
            def __init__(self):
                # Assignment parameters (needed for AssignmentService)
                self.immediate_assignment_threshold = 5.0
                self.periodic_interval = 10.0
                self.throughput_factor = 1.0
                self.age_factor = 0.1
                self.driver_speed = 0.5
                
                # Pairing configuration
                self.pairing_enabled = False
        
        return TestConfig()
    
    @pytest.fixture
    def test_environment(self):
        """Set up the common test environment."""
        env = simpy.Environment()
        event_dispatcher = EventDispatcher()
        driver_repo = DriverRepository()
        order_repo = OrderRepository()
        pair_repo = PairRepository()
        delivery_unit_repo = DeliveryUnitRepository()
        
        return {
            "env": env,
            "event_dispatcher": event_dispatcher,
            "driver_repo": driver_repo,
            "order_repo": order_repo,
            "pair_repo": pair_repo,
            "delivery_unit_repo": delivery_unit_repo
        }
    
    def test_driver_login_triggers_logout_scheduling(self, test_environment):
        """
        Test that DriverLoggedInEvent triggers the scheduling of driver logout.
        
        This test verifies that when a driver logs in, the scheduling service
        properly sets up the logout monitoring process.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        driver_repo = test_environment["driver_repo"]
        
        # Create the driver scheduling service
        scheduling_service = DriverSchedulingService(
            env=env,
            event_dispatcher=event_dispatcher,
            driver_repository=driver_repo
        )
        
        # Create a test driver
        driver = Driver(
            driver_id="D1",
            initial_location=[3, 3],
            login_time=env.now,
            service_duration=120  # 2 hours intended service
        )
        driver.entity_type = EntityType.DRIVER
        driver_repo.add(driver)
        
        # Add a spy to track logout scheduling calls
        scheduled_logouts = []
        original_schedule_method = scheduling_service.schedule_driver_logout
        
        def spy_schedule_logout(driver_id, intended_logout_time):
            scheduled_logouts.append((driver_id, intended_logout_time))
            return original_schedule_method(driver_id, intended_logout_time)
        
        scheduling_service.schedule_driver_logout = spy_schedule_logout
        
        # ACT - Dispatch driver login event
        event_dispatcher.dispatch(DriverLoggedInEvent(
            timestamp=env.now,
            driver_id=driver.driver_id,
            initial_location=driver.location,
            service_duration=driver.service_duration
        ))
        
        # Run briefly to process the event
        env.run(until=0.1)
        
        # ASSERT
        assert len(scheduled_logouts) == 1, "Logout should be scheduled for the driver"
        scheduled_driver_id, scheduled_time = scheduled_logouts[0]
        assert scheduled_driver_id == driver.driver_id, "Logout should be scheduled for correct driver"
        assert scheduled_time == driver.intended_logout_time, "Logout should be scheduled for correct time"
    
    def test_delivery_completion_with_eligible_driver_dispatches_availability_event(self, test_environment):
        """
        Test that delivery completion with an eligible driver dispatches DriverAvailableForAssignmentEvent.
        
        This test verifies the new architectural flow where only eligible drivers
        trigger assignment attempts.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        driver_repo = test_environment["driver_repo"]
        
        # Create the driver scheduling service
        scheduling_service = DriverSchedulingService(
            env=env,
            event_dispatcher=event_dispatcher,
            driver_repository=driver_repo
        )
        
        # Create a test driver who is NOT overdue
        current_time = 100.0
        env._now = current_time  # Set simulation time
        
        driver = Driver(
            driver_id="D1",
            initial_location=[3, 3],
            login_time=current_time - 60,  # Logged in 60 minutes ago
            service_duration=180  # Intends to work 180 minutes total
        )
        driver.entity_type = EntityType.DRIVER
        driver.state = DriverState.AVAILABLE  # Just finished a delivery
        driver_repo.add(driver)
        
        # Verify driver is not overdue (this is our test setup verification)
        assert current_time < driver.intended_logout_time, \
            f"Test setup error: Driver should not be overdue. Current: {current_time}, Logout: {driver.intended_logout_time}"
        
        # Track DriverAvailableForAssignmentEvent
        availability_events = []
        event_dispatcher.register(DriverAvailableForAssignmentEvent, 
                                lambda e: availability_events.append(e))
        
        # ACT - Dispatch delivery completion event
        event_dispatcher.dispatch(DeliveryUnitCompletedEvent(
            timestamp=current_time,
            delivery_unit_id="DU-O1-D1",
            driver_id=driver.driver_id
        ))
        
        # Run briefly to process the event
        env.run(until=current_time + 0.1)
        
        # ASSERT
        assert len(availability_events) == 1, "DriverAvailableForAssignmentEvent should be dispatched"
        event = availability_events[0]
        assert event.driver_id == driver.driver_id, "Event should reference the correct driver"
        assert event.timestamp == current_time, "Event should have correct timestamp"
    
    def test_delivery_completion_with_overdue_driver_logs_out_immediately(self, test_environment):
        """
        Test that delivery completion with an overdue driver logs them out immediately.
        
        This test verifies that overdue drivers don't trigger assignment attempts.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        driver_repo = test_environment["driver_repo"]
        
        # Create the driver scheduling service
        scheduling_service = DriverSchedulingService(
            env=env,
            event_dispatcher=event_dispatcher,
            driver_repository=driver_repo
        )
        
        # Create a test driver who IS overdue
        current_time = 200.0
        env._now = current_time  # Set simulation time
        
        driver = Driver(
            driver_id="D1",
            initial_location=[3, 3],
            login_time=current_time - 180,  # Logged in 180 minutes ago
            service_duration=120  # Only intended to work 120 minutes
        )
        driver.entity_type = EntityType.DRIVER
        driver.state = DriverState.AVAILABLE  # Just finished a delivery
        driver_repo.add(driver)
        
        # Verify driver IS overdue (this is our test setup verification)
        assert current_time >= driver.intended_logout_time, \
            f"Test setup error: Driver should be overdue. Current: {current_time}, Logout: {driver.intended_logout_time}"
        
        # Track events
        availability_events = []
        logout_events = []
        event_dispatcher.register(DriverAvailableForAssignmentEvent, 
                                lambda e: availability_events.append(e))
        event_dispatcher.register(DriverLoggedOutEvent, 
                                lambda e: logout_events.append(e))
        
        # ACT - Dispatch delivery completion event
        event_dispatcher.dispatch(DeliveryUnitCompletedEvent(
            timestamp=current_time,
            delivery_unit_id="DU-O1-D1",
            driver_id=driver.driver_id
        ))
        
        # Run briefly to process the event
        env.run(until=current_time + 0.1)
        
        # ASSERT
        assert len(availability_events) == 0, "No DriverAvailableForAssignmentEvent should be dispatched for overdue driver"
        assert len(logout_events) == 1, "DriverLoggedOutEvent should be dispatched"
        assert driver.state == DriverState.OFFLINE, "Driver should be logged out"
        
        logout_event = logout_events[0]
        assert logout_event.driver_id == driver.driver_id, "Logout event should reference correct driver"
    
    def test_availability_event_triggers_assignment_attempt(self, test_environment, test_config):
        """
        Test that DriverAvailableForAssignmentEvent triggers assignment attempt in AssignmentService.
        
        This test verifies the complete event flow from scheduling service to assignment service.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        driver_repo = test_environment["driver_repo"]
        order_repo = test_environment["order_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
        # Create both services
        scheduling_service = DriverSchedulingService(
            env=env,
            event_dispatcher=event_dispatcher,
            driver_repository=driver_repo
        )
        
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=config
        )
        
        # Create a test driver
        driver = Driver(
            driver_id="D1",
            initial_location=[3, 3],
            login_time=env.now,
            service_duration=120
        )
        driver.entity_type = EntityType.DRIVER
        driver.state = DriverState.AVAILABLE
        driver_repo.add(driver)
        
        # Add a spy to track assignment attempts
        assignment_attempts = []
        original_attempt_method = assignment_service.attempt_immediate_assignment_from_driver
        
        def spy_assignment_attempt(driver):
            assignment_attempts.append(driver.driver_id)
            return original_attempt_method(driver)
        
        assignment_service.attempt_immediate_assignment_from_driver = spy_assignment_attempt
        
        # ACT - Dispatch DriverAvailableForAssignmentEvent
        event_dispatcher.dispatch(DriverAvailableForAssignmentEvent(
            timestamp=env.now,
            driver_id=driver.driver_id
        ))
        
        # Run briefly to process the event
        env.run(until=0.1)
        
        # ASSERT
        assert len(assignment_attempts) == 1, "Assignment attempt should be triggered"
        assert assignment_attempts[0] == driver.driver_id, "Assignment attempt should be for correct driver"
    
    def test_scheduled_logout_with_available_driver(self, test_environment):
        """
        Test that scheduled logout works correctly when driver is available.
        
        This test verifies the SimPy process-based scheduling mechanism.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        driver_repo = test_environment["driver_repo"]
        
        # Create the driver scheduling service
        scheduling_service = DriverSchedulingService(
            env=env,
            event_dispatcher=event_dispatcher,
            driver_repository=driver_repo
        )
        
        # Create a test driver with short service duration for faster testing
        service_duration = 10.0  # 10 minutes
        driver = Driver(
            driver_id="D1",
            initial_location=[3, 3],
            login_time=env.now,
            service_duration=service_duration
        )
        driver.entity_type = EntityType.DRIVER
        driver.state = DriverState.AVAILABLE
        driver_repo.add(driver)
        
        # Track logout events
        logout_events = []
        event_dispatcher.register(DriverLoggedOutEvent, lambda e: logout_events.append(e))
        
        # Start the scheduling process
        scheduling_service.schedule_driver_logout(driver.driver_id, driver.intended_logout_time)
        
        # ACT - Run simulation until after the intended logout time
        env.run(until=service_duration + 1)
        
        # ASSERT
        assert len(logout_events) == 1, "Driver should be logged out at scheduled time"
        assert driver.state == DriverState.OFFLINE, "Driver should be in OFFLINE state"
        
        logout_event = logout_events[0]
        assert logout_event.driver_id == driver.driver_id, "Logout event should reference correct driver"
    
    def test_scheduled_logout_with_delivering_driver(self, test_environment):
        """
        Test that scheduled logout is deferred when driver is delivering.
        
        This test verifies that busy drivers don't get interrupted by logout attempts.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        driver_repo = test_environment["driver_repo"]
        
        # Create the driver scheduling service
        scheduling_service = DriverSchedulingService(
            env=env,
            event_dispatcher=event_dispatcher,
            driver_repository=driver_repo
        )
        
        # Create a test driver with short service duration
        service_duration = 5.0  # 5 minutes
        driver = Driver(
            driver_id="D1",
            initial_location=[3, 3],
            login_time=env.now,
            service_duration=service_duration
        )
        driver.entity_type = EntityType.DRIVER
        driver.state = DriverState.DELIVERING  # Driver is busy
        driver_repo.add(driver)
        
        # Track logout events
        logout_events = []
        event_dispatcher.register(DriverLoggedOutEvent, lambda e: logout_events.append(e))
        
        # Start the scheduling process
        scheduling_service.schedule_driver_logout(driver.driver_id, driver.intended_logout_time)
        
        # ACT - Run simulation until after the intended logout time
        env.run(until=service_duration + 1)
        
        # ASSERT
        assert len(logout_events) == 0, "No logout should occur while driver is delivering"
        assert driver.state == DriverState.DELIVERING, "Driver should remain in DELIVERING state"
    
    def test_complete_driver_lifecycle_integration(self, test_environment, test_config):
        """
        Test a complete driver lifecycle from login through assignment to logout.
        
        This integration test verifies the entire flow works together correctly.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        driver_repo = test_environment["driver_repo"]
        order_repo = test_environment["order_repo"]
        pair_repo = test_environment["pair_repo"]
        delivery_unit_repo = test_environment["delivery_unit_repo"]
        config = test_config
        
        # Create both services
        scheduling_service = DriverSchedulingService(
            env=env,
            event_dispatcher=event_dispatcher,
            driver_repository=driver_repo
        )
        
        assignment_service = AssignmentService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            driver_repository=driver_repo,
            pair_repository=pair_repo,
            delivery_unit_repository=delivery_unit_repo,
            config=config
        )
        
        # Create a driver and order
        driver = Driver(
            driver_id="D1",
            initial_location=[3, 3],
            login_time=env.now,
            service_duration=30  # 30 minutes service
        )
        driver.entity_type = EntityType.DRIVER
        driver_repo.add(driver)
        
        order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[5, 5],
            arrival_time=env.now
        )
        order.entity_type = EntityType.ORDER
        order_repo.add(order)
        
        # Track key events
        availability_events = []
        logout_events = []
        event_dispatcher.register(DriverAvailableForAssignmentEvent, 
                                lambda e: availability_events.append(e))
        event_dispatcher.register(DriverLoggedOutEvent, 
                                lambda e: logout_events.append(e))
        
        # ACT - Simulate the complete lifecycle
        
        # Step 1: Driver logs in
        event_dispatcher.dispatch(DriverLoggedInEvent(
            timestamp=env.now,
            driver_id=driver.driver_id,
            initial_location=driver.location,
            service_duration=driver.service_duration
        ))
        env.run(until=1)
        
        # Step 2: Simulate a delivery completion within service time
        env._now = 20  # 20 minutes into service (within 30 minute limit)

        # Manually simulate what DeliveryService would do
        driver.transition_to(DriverState.AVAILABLE, event_dispatcher, env)

        # Now dispatch the completion event
        event_dispatcher.dispatch(DeliveryUnitCompletedEvent(
            timestamp=env.now,
            delivery_unit_id="DU-O1-D1",
            driver_id=driver.driver_id
        ))
        env.run(until=21)
        
        # Step 3: Let scheduled logout time arrive
        env.run(until=35)  # Past the 30 minute service duration
        
        # ASSERT
        assert len(availability_events) == 1, "Driver should become available after delivery completion"
        assert len(logout_events) == 1, "Driver should log out at scheduled time"
        assert driver.state == DriverState.OFFLINE, "Driver should end in OFFLINE state"