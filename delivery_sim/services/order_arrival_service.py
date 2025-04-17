from delivery_sim.entities.order import Order
from delivery_sim.events.order_events import OrderCreatedEvent

class OrderArrivalService:
    """
    Service responsible for generating new orders entering the system.
    
    This service runs as a continuous SimPy process, creating new orders
    based on configured inter-arrival times and dispatching events when
    orders are created.
    """
    
    def __init__(self, env, event_dispatcher, order_repository, config, id_generator):
        """
        Initialize the order arrival service.
        
        Args:
            env: SimPy environment
            event_dispatcher: Central event dispatcher
            order_repository: Repository for storing created orders
            config: Configuration containing arrival rate parameters
            id_generator: Generator for unique order IDs
        """
        self.env = env
        self.event_dispatcher = event_dispatcher
        self.order_repository = order_repository
        self.config = config
        self.id_generator = id_generator
        
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
        Generate the time until the next order arrival.
        
        In a complete implementation, this would use a random distribution.
        
        Returns:
            float: Time until next arrival in minutes
        """
        # Placeholder: In a real implementation, this would use a distribution
        return self.config.mean_order_inter_arrival_time
    
    def _select_restaurant_location(self):
        """
        Select a restaurant location for a new order.
        
        In a complete implementation, this would select from registered restaurants.
        
        Returns:
            list: [x, y] coordinates of restaurant
        """
        # Placeholder: In a real implementation, this would select from restaurants
        return [0, 0]
    
    def _generate_customer_location(self):
        """
        Generate a customer location for a new order.
        
        In a complete implementation, this would use a spatial distribution.
        
        Returns:
            list: [x, y] coordinates of customer
        """
        # Placeholder: In a real implementation, this would use a distribution
        return [5, 5]