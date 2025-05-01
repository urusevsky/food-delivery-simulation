from delivery_sim.entities.order import Order
from delivery_sim.events.order_events import OrderCreatedEvent

class OrderArrivalService:
    """
    Service responsible for generating new orders entering the system.
    
    This service runs as a continuous SimPy process, creating new orders
    based on configured inter-arrival times and dispatching events when
    orders are created.
    """
    
    def __init__(self, env, event_dispatcher, order_repository, restaurant_repository, config, id_generator, operational_rng_manager):
        """
        Initialize the order arrival service.
        
        Args:
            env: SimPy environment
            event_dispatcher: Central event dispatcher
            order_repository: Repository for storing created orders
            restaurant_repository: Repository for restaurant selection
            config: Configuration containing arrival rate parameters
            id_generator: Generator for unique order IDs
            operational_rng_manager: Manager for random number streams
        """
        self.env = env
        self.event_dispatcher = event_dispatcher
        self.order_repository = order_repository
        self.restaurant_repository = restaurant_repository  # Added for restaurant selection
        self.config = config
        self.id_generator = id_generator
        
        # Get all random streams at initialization time
        self.arrival_stream = operational_rng_manager.get_stream('order_arrivals')
        self.location_stream = operational_rng_manager.get_stream('customer_locations')
        self.restaurant_selection_stream = operational_rng_manager.get_stream('restaurant_selection')
        
        # Start the arrival process
        self.process = env.process(self._arrival_process())
    
    def _arrival_process(self):
        """SimPy process that generates new orders at configured intervals."""
        while True:
            # Generate time until next order arrival
            inter_arrival_time = self._generate_inter_arrival_time()
            yield self.env.timeout(inter_arrival_time)
            
            # Generate order attributes
            order_id = self.id_generator.next()
            restaurant_location = self._select_restaurant_location()
            customer_location = self._generate_customer_location()
            
            # Create new order
            new_order = Order(
                order_id=order_id,
                restaurant_location=restaurant_location,
                customer_location=customer_location,
                arrival_time=self.env.now
            )
            
            # Add to repository
            self.order_repository.add(new_order)
            
            # Dispatch order created event
            self.event_dispatcher.dispatch(OrderCreatedEvent(
                timestamp=self.env.now,
                order_id=order_id,
                restaurant_location=restaurant_location,
                customer_location=customer_location
            ))
            
            # Log for debugging
            print(f"Order {order_id} created at time {self.env.now}")
    
    def _generate_inter_arrival_time(self):
        """
        Generate the time until the next order arrival using an exponential distribution.
        
        This models arrivals as a Poisson process, which is standard for independent
        arrivals in service systems.
        
        Returns:
            float: Time until next arrival in minutes
        """
        return self.arrival_stream.exponential(self.config.mean_order_inter_arrival_time)

    def _select_restaurant_location(self):
        """
        Select a restaurant location for a new order.
        
        This randomly selects from the existing restaurants in the system.
        
        Returns:
            list: [x, y] coordinates of restaurant
        """
        # Get all restaurants from the repository
        restaurants = self.restaurant_repository.find_all()
        
        # Randomly select one
        selected_restaurant = self.restaurant_selection_stream.choice(restaurants)
        
        return selected_restaurant.location

    def _generate_customer_location(self):
        """
        Generate a customer location for a new order.
        
        This uses a uniform distribution across the delivery area.
        In a more sophisticated model, this might use hotspots or other spatial distributions.
        
        Returns:
            list: [x, y] coordinates of customer
        """
        area_size = self.config.delivery_area_size
        return self.location_stream.uniform(0, area_size, size=2).tolist()