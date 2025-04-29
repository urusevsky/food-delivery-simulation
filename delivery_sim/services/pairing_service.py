from delivery_sim.events.order_events import OrderCreatedEvent
from delivery_sim.events.pair_events import PairCreatedEvent, PairingFailedEvent
from delivery_sim.entities.pair import Pair
from delivery_sim.entities.states import OrderState
from delivery_sim.utils.location_utils import calculate_distance, locations_are_equal

class PairingService:
    """
    Service responsible for evaluating pairing opportunities between orders.
    
    This service responds to new order arrivals by checking if they can be
    paired with existing unassigned orders based on proximity criteria.
    """
    
    def __init__(self, env, event_dispatcher, order_repository, pair_repository, config):
        """
        Initialize the pairing service with its dependencies.
        
        Args:
            env: SimPy environment
            event_dispatcher: Central event dispatcher
            order_repository: Repository for accessing orders
            pair_repository: Repository for storing created pairs
            config: Configuration containing pairing parameters
        """
        self.env = env
        self.event_dispatcher = event_dispatcher
        self.order_repository = order_repository
        self.pair_repository = pair_repository
        self.config = config
        
        # Register for events this service handles
        self.event_dispatcher.register(OrderCreatedEvent, self.handle_order_created)
    
    # ===== Event Handlers =====
    
    def handle_order_created(self, event):
        """
        Handler for OrderCreatedEvent. Attempts to pair the new order.
        
        This is a lightweight handler that extracts the necessary information
        and delegates to the appropriate operation.
        
        Args:
            event: The OrderCreatedEvent
        """
        order_id = event.order_id
        self.attempt_pairing(order_id)
    
    # ===== Operations =====
    
    def attempt_pairing(self, order_id):
        """
        Try to pair the new order with an existing order.
        
        This is the main operation that:
        1. Finds potential pairing candidates 
        2. Evaluates the best match based on delivery cost
        3. Forms a pair if a good match is found
        
        Args:
            order_id: ID of the order to attempt pairing for
            
        Returns:
            bool: True if pairing succeeded, False otherwise
        """
        # Get the order from repository
        new_order = self.order_repository.find_by_id(order_id)
        
        # Find potential candidates for pairing
        candidates = self.find_pairing_candidates(new_order)
        if not candidates:
            # Business outcome: No suitable pairing candidates found
            self.event_dispatcher.dispatch(PairingFailedEvent(
                timestamp=self.env.now,
                order_id=order_id
            ))
            return False
        
        # Find the best match among candidates
        # (This will always succeed if candidates exist)
        best_match, best_sequence, best_cost = self.calculate_best_match(new_order, candidates)
        
        # Form the pair with the best match
        self.form_pair(new_order, best_match, best_sequence, best_cost)
        return True
    
    def find_pairing_candidates(self, new_order):
        """
        Find orders that meet proximity criteria for pairing.
        
        Args:
            new_order: The order to find candidates for
            
        Returns:
            list: Orders that meet proximity criteria
        """
        # Get all unassigned single orders
        all_singles = self.order_repository.find_unassigned_orders()
        
        # Filter out the new order itself
        potential_candidates = [
            order for order in all_singles 
            if order.order_id != new_order.order_id
        ]
        
        # Apply proximity filters
        return [
            candidate for candidate in potential_candidates
            if self._check_proximity_constraints(new_order, candidate)
        ]
    
    def calculate_best_match(self, new_order, candidates):
        """
        Find the best order to pair with and the optimal delivery sequence.
        
        Args:
            new_order: The new order to pair
            candidates: List of potential pairing candidates
            
        Returns:
            tuple: (best_candidate, best_sequence, best_cost) for the optimal pairing
        """
        best_candidate = None
        best_sequence = None
        best_cost = float('inf')
        
        for candidate in candidates:
            sequence, cost = self.evaluate_sequences(new_order, candidate)
            if cost < best_cost:
                best_cost = cost
                best_sequence = sequence
                best_candidate = candidate
        
        return best_candidate, best_sequence, best_cost
    
    def form_pair(self, order1, order2, sequence, cost):
        """
        Create a pair from two orders and update their state.
        
        Args:
            order1: First order in the pair
            order2: Second order in the pair
            sequence: Optimal delivery sequence for this pair
            cost: Total delivery cost for this pair
            
        Returns:
            Pair: The newly created pair
        """
        # Create the pair entity
        pair = Pair(order1, order2, self.env.now)
        pair.optimal_sequence = sequence
        pair.optimal_cost = cost
        
        # Add to repository
        self.pair_repository.add(pair)
        
        # Update order states
        order1.transition_to(OrderState.PAIRED, self.event_dispatcher, self.env)
        order2.transition_to(OrderState.PAIRED, self.event_dispatcher, self.env)
        
        # Dispatch pair created event
        self.event_dispatcher.dispatch(PairCreatedEvent(
            timestamp=self.env.now,
            pair_id=pair.pair_id,
            order1_id=order1.order_id,
            order2_id=order2.order_id
        ))
        
        # Log for debugging
        print(f"Formed pair {pair.pair_id} at time {self.env.now}")
        
        return pair
    
    # ===== Helper Methods =====
    
    def evaluate_sequences(self, order1, order2):
        """
        Evaluate all possible delivery sequences for a pair of orders.
        
        Args:
            order1: First order to evaluate
            order2: Second order to evaluate
            
        Returns:
            tuple: (best_sequence, best_cost) for the optimal delivery sequence
        """
        sequences = []
        same_restaurant = locations_are_equal(
            order1.restaurant_location, 
            order2.restaurant_location
        )

        if not same_restaurant:
            # Different restaurants - consider all possible sequences
            sequences.append([
                order1.restaurant_location, order2.restaurant_location, 
                order1.customer_location, order2.customer_location
            ])
            sequences.append([
                order1.restaurant_location, order2.restaurant_location, 
                order2.customer_location, order1.customer_location
            ])
            sequences.append([
                order2.restaurant_location, order1.restaurant_location, 
                order1.customer_location, order2.customer_location
            ])
            sequences.append([
                order2.restaurant_location, order1.restaurant_location, 
                order2.customer_location, order1.customer_location
            ])
        else:
            # Same restaurant - only need to determine customer visit order
            sequences.append([
                order1.restaurant_location,
                order1.customer_location, 
                order2.customer_location
            ])
            sequences.append([
                order1.restaurant_location,
                order2.customer_location, 
                order1.customer_location
            ])
        
        best_cost = float('inf')
        best_sequence = None
        
        for seq in sequences:
            cost = self.calculate_travel_distance(seq)
            if cost < best_cost:
                best_cost = cost
                best_sequence = seq
        
        return best_sequence, best_cost
    
    def calculate_travel_distance(self, sequence):
        """
        Calculate total travel distance for a delivery sequence.
        
        Args:
            sequence: List of locations to visit in order
            
        Returns:
            float: Total distance to travel the sequence
        """
        total_distance = 0.0
        for i in range(len(sequence) - 1):
            total_distance += calculate_distance(sequence[i], sequence[i + 1])
        return total_distance
    
    def _check_proximity_constraints(self, order1, order2):
        """
        Check if two orders satisfy proximity constraints for pairing.
        
        Args:
            order1: First order to check
            order2: Second order to check
            
        Returns:
            bool: True if orders meet proximity criteria, False otherwise
        """
        restaurant_distance = calculate_distance(
            order1.restaurant_location,
            order2.restaurant_location
        )
        customer_distance = calculate_distance(
            order1.customer_location,
            order2.customer_location
        )
        
        return (restaurant_distance <= self.config.restaurants_proximity_threshold and
                customer_distance <= self.config.customers_proximity_threshold)
    
