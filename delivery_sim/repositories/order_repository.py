from delivery_sim.entities.states import OrderState

class OrderRepository:
    """
    Repository for managing orders in the simulation.
    
    This repository stores all orders created during the simulation and
    provides methods for finding orders based on various criteria.
    """
    
    def __init__(self):
        """Initialize an empty order repository."""
        self.orders = {}  # Maps order_id to Order objects
    
    def add(self, order):
        """
        Add an order to the repository.
        
        Args:
            order: The Order object to add
        """
        self.orders[order.order_id] = order
    
    def find_by_id(self, order_id):
        """
        Find an order by its ID.
        
        Args:
            order_id: The ID of the order to find
            
        Returns:
            Order: The found order or None if not found
        """
        return self.orders.get(order_id)
    
    def find_all(self):
        """
        Get all orders in the repository.
        
        Returns:
            list: All Order objects in the repository
        """
        return list(self.orders.values())
    
    def find_by_state(self, state):
        """
        Find all orders in a specific state.
        
        Args:
            state: The OrderState to filter by
            
        Returns:
            list: Order objects with the specified state
        """
        return [order for order in self.orders.values() if order.state == state]
    
    def find_waiting_orders(self):
        """
        Find orders waiting for assignment (in CREATED or PAIRED state).
        
        Returns:
            list: Orders waiting for driver assignment
        """
        return [order for order in self.orders.values() 
                if order.state in [OrderState.CREATED, OrderState.PAIRED]]
    
    def count(self):
        """
        Get the total number of orders in the repository.
        
        Returns:
            int: The number of orders
        """
        return len(self.orders)