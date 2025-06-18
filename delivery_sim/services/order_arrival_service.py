from delivery_sim.entities.order import Order
from delivery_sim.events.order_events import OrderCreatedEvent
from delivery_sim.utils.logging_system import get_logger
from delivery_sim.utils.location_utils import format_location

class OrderArrivalService:
    """
    Service responsible for generating new orders entering the system.
    
    This service runs as a continuous SimPy process, creating new orders
    based on configured inter-arrival times and dispatching events when
    orders are created.
    """
    
    def __init__(self, env, event_dispatcher, order_repository, restaurant_repository, config, id_generator, operational_rng_manager):
        """Initialize the order arrival service."""
        # Get a logger instance specific to this component
        self.logger = get_logger("services.order_arrival")
        
        # Store dependencies
        self.env = env
        self.event_dispatcher = event_dispatcher
        self.order_repository = order_repository
        self.restaurant_repository = restaurant_repository
        self.config = config
        self.id_generator = id_generator
        
        # Get all random streams at initialization time
        self.arrival_stream = operational_rng_manager.get_stream('order_arrivals')
        self.location_stream = operational_rng_manager.get_stream('customer_locations')
        self.restaurant_selection_stream = operational_rng_manager.get_stream('restaurant_selection')
        
        # Log service initialization with configuration details
        self.logger.info(f"[t={self.env.now:.2f}] OrderArrivalService initialized with mean inter-arrival time: {config.mean_order_inter_arrival_time} minutes")
        
        # Start the arrival process - use INFO level for important system milestone
        self.logger.info(f"[t={self.env.now:.2f}] Starting order arrival process")
        self.process = env.process(self._arrival_process())
    
    def _arrival_process(self):
        """SimPy process that generates new orders at configured intervals."""
        while True:
            # Generate time until next order arrival
            inter_arrival_time = self._generate_inter_arrival_time()
            self.logger.debug(f"[t={self.env.now:.2f}] Next order will arrive in {inter_arrival_time:.2f} minutes")
            
            yield self.env.timeout(inter_arrival_time)
            
            # Generate order attributes
            order_id = self.id_generator.next()
            restaurant_location = self._select_restaurant_location()
            customer_location = self._generate_customer_location()
            
            self.logger.debug(f"[t={self.env.now:.2f}] Generated attributes for order {order_id}: "
                            f"restaurant at {format_location(restaurant_location)}, customer at {format_location(customer_location)}")

            
            # Create new order
            new_order = Order(
                order_id=order_id,
                restaurant_location=restaurant_location,
                customer_location=customer_location,
                arrival_time=self.env.now
            )
            
            # Add to repository
            self.order_repository.add(new_order)
            
            # Log order creation
            self.logger.info(f"[t={self.env.now:.2f}] Created order {order_id} from restaurant at {format_location(restaurant_location)} to customer at {format_location(customer_location)}")
            
            # Dispatch order created event
            self.logger.simulation_event(f"[t={self.env.now:.2f}] Dispatching OrderCreatedEvent for order {order_id}")
            self.event_dispatcher.dispatch(OrderCreatedEvent(
                timestamp=self.env.now,
                order_id=order_id,
                restaurant_id=0,  # This appears to be missing in the current implementation
                restaurant_location=restaurant_location,
                customer_location=customer_location
            ))
    
    def _generate_inter_arrival_time(self):
        """Generate the time until the next order arrival using an exponential distribution."""
        return self.arrival_stream.exponential(self.config.mean_order_inter_arrival_time)

    def _select_restaurant_location(self):
        """Select a restaurant location for a new order."""
        # Get all restaurants from the repository
        restaurants = self.restaurant_repository.find_all()
        
        # Randomly select one
        selected_restaurant = self.restaurant_selection_stream.choice(restaurants)
        
        self.logger.debug(f"[t={self.env.now:.2f}] Selected restaurant {selected_restaurant.restaurant_id} at {format_location(selected_restaurant.location)}")
        return selected_restaurant.location

    def _generate_customer_location(self):
        """Generate a customer location for a new order."""
        area_size = self.config.delivery_area_size
        location = self.location_stream.uniform(0, area_size, size=2).tolist()
        return location