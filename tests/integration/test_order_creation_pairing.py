# tests/integration/test_order_creation_pairing.py
import pytest
import simpy
from unittest.mock import patch, Mock

# Import core components
from delivery_sim.events.event_dispatcher import EventDispatcher
from delivery_sim.events.order_events import OrderCreatedEvent
from delivery_sim.events.pair_events import PairCreatedEvent, PairingFailedEvent
from delivery_sim.repositories.order_repository import OrderRepository
from delivery_sim.repositories.pair_repository import PairRepository
from delivery_sim.repositories.restaurant_repository import RestaurantRepository
from delivery_sim.services.order_arrival_service import OrderArrivalService
from delivery_sim.services.pairing_service import PairingService
from delivery_sim.simulation.rng_manager import OperationalRNGManager
from delivery_sim.entities.restaurant import Restaurant
from delivery_sim.entities.order import Order
from delivery_sim.entities.states import OrderState
from delivery_sim.utils.id_generator import PrefixedIdGenerator
from delivery_sim.utils.location_utils import calculate_distance


class TestOrderCreationPairing:
    """
    Integration tests for OrderArrivalService and PairingService interaction.
    
    These tests verify:
    1. OrderCreatedEvent triggers the pairing attempt
    2. Pairing succeeds with compatible orders
    3. Pairing fails with incompatible orders
    4. OrderArrivalService correctly dispatches OrderCreatedEvent
    """
    
    @pytest.fixture
    def test_config(self):
        """Create a simple test configuration."""
        class TestConfig:
            def __init__(self):
                # OrderArrivalService parameters
                self.mean_order_inter_arrival_time = 1.0
                self.delivery_area_size = 10.0
                
                # PairingService parameters
                self.pairing_enabled = True
                self.restaurants_proximity_threshold = 3.0
                self.customers_proximity_threshold = 4.0
        
        return TestConfig()
    
    @pytest.fixture
    def test_environment(self, test_config):
        """Set up the complete test environment."""
        env = simpy.Environment()
        event_dispatcher = EventDispatcher()
        order_repo = OrderRepository()
        pair_repo = PairRepository()
        restaurant_repo = RestaurantRepository()
        
        # Add test restaurants
        restaurant1 = Restaurant("R1", [3, 3])
        restaurant2 = Restaurant("R2", [4, 4])  # 1.4km from restaurant1
        restaurant3 = Restaurant("R3", [9, 9])  # Far from others (8.5km from restaurant1)
        restaurant_repo.add(restaurant1)
        restaurant_repo.add(restaurant2)
        restaurant_repo.add(restaurant3)
        
        # Create ID generator
        id_generator = PrefixedIdGenerator("O")
        
        # Create RNG manager
        operational_rng = OperationalRNGManager(42, 0)
        
        # Return all components
        return {
            "env": env,
            "event_dispatcher": event_dispatcher,
            "order_repo": order_repo,
            "pair_repo": pair_repo,
            "restaurant_repo": restaurant_repo,
            "id_generator": id_generator,
            "operational_rng": operational_rng,
            "config": test_config
        }
    
    def test_order_created_event_triggers_pairing_attempt(self, test_environment):
        """
        Test that OrderCreatedEvent triggers a pairing attempt.
        
        This test verifies the integration between the event system
        and PairingService by tracking if attempt_pairing gets called.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        pair_repo = test_environment["pair_repo"]
        config = test_environment["config"]
        
        # Create the pairing service
        pairing_service = PairingService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            pair_repository=pair_repo,
            config=config
        )
        
        # Add a spy to track pairing attempts
        pairing_attempts = []
        original_attempt_pairing = pairing_service.attempt_pairing
        
        def spy_attempt_pairing(order):
            pairing_attempts.append(order.order_id)
            return original_attempt_pairing(order)
        
        pairing_service.attempt_pairing = spy_attempt_pairing
        
        # Create a test order
        new_order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[7, 7],
            arrival_time=env.now
        )
        order_repo.add(new_order)
        
        # ACT - Manually dispatch the event
        event_dispatcher.dispatch(OrderCreatedEvent(
            timestamp=env.now,
            order_id=new_order.order_id,
            restaurant_id="R1",
            restaurant_location=new_order.restaurant_location,
            customer_location=new_order.customer_location
        ))
        
        # Run briefly to process events
        env.run(until=0.1)
        
        # ASSERT
        assert len(pairing_attempts) == 1, "PairingService should attempt to pair the order"
        assert pairing_attempts[0] == new_order.order_id, "Pairing attempt should be for the correct order ID"
    
    def test_successful_pairing_with_compatible_orders(self, test_environment):
        """
        Test successful pairing when two compatible orders exist.
        
        This test verifies that when an OrderCreatedEvent is dispatched for an
        order that has a compatible match, both orders are paired successfully.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        pair_repo = test_environment["pair_repo"]
        config = test_environment["config"]
        
        # Create the pairing service
        pairing_service = PairingService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            pair_repository=pair_repo,
            config=config
        )
        
        # Create first order that will already be in the system
        existing_order = Order(
            order_id="O1",
            restaurant_location=[3, 3],
            customer_location=[6, 6],
            arrival_time=env.now
        )
        order_repo.add(existing_order)
        
        # Create second order that should pair with the first
        new_order = Order(
            order_id="O2",
            restaurant_location=[3, 3],  # Same restaurant
            customer_location=[7, 7],    # Close to first customer (distance ~1.4km, within threshold)
            arrival_time=env.now
        )
        order_repo.add(new_order)
        
        # Verify the distance between customers is within threshold
        customer_distance = calculate_distance(existing_order.customer_location, new_order.customer_location)
        assert customer_distance < config.customers_proximity_threshold, \
               f"Test setup error: Customer distance ({customer_distance:.2f}km) exceeds threshold ({config.customers_proximity_threshold}km)"
        
        # ACT - Manually dispatch event for new order
        event_dispatcher.dispatch(OrderCreatedEvent(
            timestamp=env.now,
            order_id=new_order.order_id,
            restaurant_id="R1",
            restaurant_location=new_order.restaurant_location,
            customer_location=new_order.customer_location
        ))
        
        # Run briefly to process events
        env.run(until=0.1)
        
        # ASSERT
        # Verify both orders are in PAIRED state
        assert existing_order.state == OrderState.PAIRED, "Existing order should be paired"
        assert new_order.state == OrderState.PAIRED, "New order should be paired"
        
        # Verify a pair was created
        assert len(pair_repo.find_all()) == 1, "A pair should be created"
        created_pair = pair_repo.find_all()[0]
        
        # Verify the pair contains both orders
        assert ((created_pair.order1 is existing_order and created_pair.order2 is new_order) or
                (created_pair.order1 is new_order and created_pair.order2 is existing_order)), \
               "The pair should contain both orders"
        
        # Verify both orders reference the pair
        assert existing_order.pair is created_pair, "Existing order should reference the pair"
        assert new_order.pair is created_pair, "New order should reference the pair"
    
    def test_pairing_failed_event_on_incompatible_orders(self, test_environment):
        """
        Test pairing failure when no compatible orders exist.
        
        This test verifies that when an OrderCreatedEvent is dispatched for an
        order with no compatible matches, a PairingFailedEvent is generated.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        pair_repo = test_environment["pair_repo"]
        config = test_environment["config"]
        
        # Create the pairing service
        pairing_service = PairingService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            pair_repository=pair_repo,
            config=config
        )
        
        # Create an order in the system that is NOT compatible
        existing_order = Order(
            order_id="O1",
            restaurant_location=[9, 9],  # Using restaurant3 location
            customer_location=[8, 8],    # Within delivery area but far from test area
            arrival_time=env.now
        )
        order_repo.add(existing_order)
        
        # Create a test order that can't pair with existing one
        new_order = Order(
            order_id="O2",
            restaurant_location=[3, 3],  # Using restaurant1 location
            customer_location=[7, 7],    # Within delivery area
            arrival_time=env.now
        )
        order_repo.add(new_order)
        
        # Verify that restaurant distance exceeds threshold
        restaurant_distance = calculate_distance(existing_order.restaurant_location, new_order.restaurant_location)
        assert restaurant_distance > config.restaurants_proximity_threshold, \
               f"Test setup error: Restaurant distance ({restaurant_distance:.2f}km) does not exceed threshold ({config.restaurants_proximity_threshold}km)"
        
        # Track PairingFailedEvents
        pairing_failed_events = []
        event_dispatcher.register(PairingFailedEvent, lambda e: pairing_failed_events.append(e))
        
        # ACT - Manually dispatch event for test order
        event_dispatcher.dispatch(OrderCreatedEvent(
            timestamp=env.now,
            order_id=new_order.order_id,
            restaurant_id="R1",
            restaurant_location=new_order.restaurant_location,
            customer_location=new_order.customer_location
        ))
        
        # Run briefly to process events
        env.run(until=0.1)
        
        # ASSERT
        # Verify both orders remain in CREATED state
        assert existing_order.state == OrderState.CREATED, "Existing order should remain unpaired"
        assert new_order.state == OrderState.CREATED, "New order should remain unpaired"
        
        # Verify no pairs were created
        assert len(pair_repo.find_all()) == 0, "No pairs should be created"
        
        # Verify a PairingFailedEvent was dispatched
        assert len(pairing_failed_events) == 1, "A PairingFailedEvent should be dispatched"
        assert pairing_failed_events[0].order_id == new_order.order_id, "Event should reference the correct order"
    
    def test_order_arrival_service_dispatches_order_created_events(self, test_environment):
        """
        Test that OrderArrivalService correctly dispatches OrderCreatedEvent.
        
        This test verifies that the real OrderArrivalService generates orders
        and dispatches the correct events that would trigger the pairing process.
        """
        # ARRANGE
        env = test_environment["env"]
        event_dispatcher = test_environment["event_dispatcher"]
        order_repo = test_environment["order_repo"]
        restaurant_repo = test_environment["restaurant_repo"]
        id_generator = test_environment["id_generator"]
        operational_rng = test_environment["operational_rng"]
        config = test_environment["config"]
        
        # Create the OrderArrivalService
        order_arrival_service = OrderArrivalService(
            env=env,
            event_dispatcher=event_dispatcher,
            order_repository=order_repo,
            restaurant_repository=restaurant_repo,
            config=config,
            id_generator=id_generator,
            operational_rng_manager=operational_rng
        )
        
        # Create a spy to track event dispatches
        dispatched_events = []
        original_dispatch = event_dispatcher.dispatch
        
        def spy_dispatch(event):
            dispatched_events.append(event)
            return original_dispatch(event)
        
        event_dispatcher.dispatch = spy_dispatch
        
        # ACT - Run simulation long enough to guarantee some orders
        # With mean arrival time of 1.0, running for 10 time units should
        # generate several orders with very high probability (>99.99%)
        env.run(until=10.0)
        
        # ASSERT
        # Filter for just OrderCreatedEvents
        order_created_events = [e for e in dispatched_events if isinstance(e, OrderCreatedEvent)]
        
        # Verify events were dispatched
        assert len(order_created_events) > 0, "OrderArrivalService should dispatch OrderCreatedEvents"
        
        # Verify orders were created in the repository
        orders = order_repo.find_all()
        assert len(orders) > 0, "Orders should be created in the repository"
        
        # Verify each event corresponds to an order in the repository
        for event in order_created_events:
            order = order_repo.find_by_id(event.order_id)
            assert order is not None, "Event should reference an existing order"
            assert order.restaurant_location == event.restaurant_location, \
                "Event should have correct restaurant location"
            assert order.customer_location == event.customer_location, \
                "Event should have correct customer location"