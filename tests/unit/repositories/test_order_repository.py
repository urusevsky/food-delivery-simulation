
import pytest
from delivery_sim.repositories.order_repository import OrderRepository
from delivery_sim.entities.order import Order
from delivery_sim.entities.states import OrderState

# Test Group 1: Basic repository operations
def test_repository_initialization():
    """
    Test that a new repository starts empty with proper initialization.
    This ensures our repository has a clean starting state.
    """
    # ARRANGE & ACT
    repository = OrderRepository()
    
    # ASSERT
    assert hasattr(repository, 'orders'), "Repository should have an orders dictionary"
    assert isinstance(repository.orders, dict), "Orders storage should be a dictionary"
    assert len(repository.orders) == 0, "New repository should start empty"
    assert repository.count() == 0, "Count should return 0 for empty repository"

def test_add_order():
    """
    Test that orders can be added to the repository.
    This is the fundamental operation that enables all other functionality.
    """
    # ARRANGE
    repository = OrderRepository()
    order = Order("123", [0, 0], [2, 3], 100)
    
    # ACT
    repository.add(order)
    
    # ASSERT
    assert repository.count() == 1, "Repository should contain one order"
    assert "123" in repository.orders, "Order ID should be in the repository"
    assert repository.orders["123"] is order, "Repository should store the exact order object"

def test_add_multiple_orders():
    """
    Test that multiple orders can be added and stored correctly.
    This verifies the repository can handle multiple entities properly.
    """
    # ARRANGE
    repository = OrderRepository()
    order1 = Order("101", [0, 0], [1, 1], 100)
    order2 = Order("102", [1, 1], [2, 2], 105)
    order3 = Order("103", [2, 2], [3, 3], 110)
    
    # ACT
    repository.add(order1)
    repository.add(order2)
    repository.add(order3)
    
    # ASSERT
    assert repository.count() == 3, "Repository should contain three orders"
    assert all(order_id in repository.orders for order_id in ["101", "102", "103"])
    
# Test Group 2: Finding orders
def test_find_by_id_existing_order():
    """
    Test that an existing order can be found by its ID.
    This is crucial for retrieving specific orders during simulation.
    """
    # ARRANGE
    repository = OrderRepository()
    order = Order("123", [0, 0], [2, 3], 100)
    repository.add(order)
    
    # ACT
    found_order = repository.find_by_id("123")
    
    # ASSERT
    assert found_order is order, "Should return the exact order object"
    assert found_order.order_id == "123", "Found order should have correct ID"

def test_find_by_id_nonexistent_order():
    """
    Test that find_by_id returns None for nonexistent orders.
    This handles the case where an order ID doesn't exist.
    """
    # ARRANGE
    repository = OrderRepository()
    order = Order("123", [0, 0], [2, 3], 100)
    repository.add(order)
    
    # ACT
    found_order = repository.find_by_id("999")  # Nonexistent ID
    
    # ASSERT
    assert found_order is None, "Should return None for nonexistent order"

def test_find_all():
    """
    Test that find_all returns all orders in the repository.
    This is useful for global operations or statistics.
    """
    # ARRANGE
    repository = OrderRepository()
    order1 = Order("101", [0, 0], [1, 1], 100)
    order2 = Order("102", [1, 1], [2, 2], 105)
    
    repository.add(order1)
    repository.add(order2)
    
    # ACT
    all_orders = repository.find_all()
    
    # ASSERT
    assert len(all_orders) == 2, "Should return all orders"
    assert order1 in all_orders, "Should include first order"
    assert order2 in all_orders, "Should include second order"
    # Verify it's a list, not the internal dictionary
    assert isinstance(all_orders, list), "Should return a list"

def test_find_all_empty_repository():
    """
    Test that find_all returns empty list for empty repository.
    This ensures the method handles the empty case correctly.
    """
    # ARRANGE
    repository = OrderRepository()
    
    # ACT
    all_orders = repository.find_all()
    
    # ASSERT
    assert all_orders == [], "Should return empty list for empty repository"
    assert isinstance(all_orders, list), "Should return a list even when empty"

# Test Group 3: Finding by state
def test_find_by_state():
    """
    Test that orders can be filtered by their state.
    This is essential for finding orders at specific stages of processing.
    """
    # ARRANGE
    repository = OrderRepository()
    
    # Create orders in different states
    order1 = Order("101", [0, 0], [1, 1], 100)  # CREATED by default
    order2 = Order("102", [1, 1], [2, 2], 105)
    order3 = Order("103", [2, 2], [3, 3], 110)
    
    # Transition order2 to PAIRED state
    order2.state = OrderState.PAIRED
    
    # Transition order3 to ASSIGNED state  
    order3.state = OrderState.ASSIGNED
    
    repository.add(order1)
    repository.add(order2)
    repository.add(order3)
    
    # ACT
    created_orders = repository.find_by_state(OrderState.CREATED)
    paired_orders = repository.find_by_state(OrderState.PAIRED)
    assigned_orders = repository.find_by_state(OrderState.ASSIGNED)
    delivered_orders = repository.find_by_state(OrderState.DELIVERED)
    
    # ASSERT
    assert len(created_orders) == 1, "Should find one CREATED order"
    assert order1 in created_orders, "Should find the correct CREATED order"
    
    assert len(paired_orders) == 1, "Should find one PAIRED order"
    assert order2 in paired_orders, "Should find the correct PAIRED order"
    
    assert len(assigned_orders) == 1, "Should find one ASSIGNED order"
    assert order3 in assigned_orders, "Should find the correct ASSIGNED order"
    
    assert len(delivered_orders) == 0, "Should find no DELIVERED orders"

# Test Group 4: Specialized queries
def test_find_unassigned_orders_returns_only_single_orders():
    """
    Test that find_unassigned_orders returns only single orders (CREATED state).
    
    This test verifies that orders in PAIRED state are NOT returned, as they
    are no longer eligible for individual assignment - their pair is the
    assignment unit.
    """
    # ARRANGE
    repository = OrderRepository()
    
    # Create orders in various states
    single_order1 = Order("101", [0, 0], [1, 1], 100)  # CREATED (single)
    single_order2 = Order("102", [1, 1], [2, 2], 105)  # CREATED (single)
    paired_order = Order("103", [2, 2], [3, 3], 110)
    assigned_order = Order("104", [3, 3], [4, 4], 115)
    
    # Move orders to different states
    paired_order.state = OrderState.PAIRED  # Part of a pair
    assigned_order.state = OrderState.ASSIGNED  # Already assigned
    
    repository.add(single_order1)
    repository.add(single_order2)
    repository.add(paired_order)
    repository.add(assigned_order)
    
    # ACT
    unassigned_single_orders = repository.find_unassigned_orders()
    
    # ASSERT
    assert len(unassigned_single_orders) == 2, "Should find only single orders"
    assert single_order1 in unassigned_single_orders
    assert single_order2 in unassigned_single_orders
    assert paired_order not in unassigned_single_orders, "Paired orders should not be included"
    assert assigned_order not in unassigned_single_orders, "Assigned orders should not be included"

# Test Group 5: Edge cases
def test_duplicate_order_ids():
    """
    Test what happens when trying to add an order with duplicate ID.
    This verifies that the repository handles ID conflicts.
    """
    # ARRANGE
    repository = OrderRepository()
    order1 = Order("123", [0, 0], [1, 1], 100)
    order2 = Order("123", [2, 2], [3, 3], 105)  # Same ID!
    
    # ACT
    repository.add(order1)
    repository.add(order2)  # This will overwrite order1
    
    # ASSERT
    assert repository.count() == 1, "Should still have only one order"
    found_order = repository.find_by_id("123")
    assert found_order is order2, "Later order should overwrite earlier one"
    assert found_order.restaurant_location == [2, 2], "Should have order2's data"

def test_repository_isolation():
    """
    Test that find_all returns a new list, not the internal storage.
    This ensures external code can't accidentally modify the repository.
    """
    # ARRANGE
    repository = OrderRepository()
    order = Order("123", [0, 0], [1, 1], 100)
    repository.add(order)
    
    # ACT
    all_orders = repository.find_all()
    all_orders.clear()  # Try to clear the returned list
    
    # ASSERT
    assert repository.count() == 1, "Repository should still contain the order"
    assert len(repository.find_all()) == 1, "Order should still be in repository"

def test_count_accuracy():
    """
    Test that count() accurately reflects the repository size.
    This is important for monitoring and statistics.
    """
    # ARRANGE
    repository = OrderRepository()
    
    # ACT & ASSERT - Test at different sizes
    assert repository.count() == 0, "Empty repository should have count 0"
    
    repository.add(Order("1", [0, 0], [1, 1], 100))
    assert repository.count() == 1, "Should count 1 after adding one order"
    
    repository.add(Order("2", [0, 0], [1, 1], 100))
    repository.add(Order("3", [0, 0], [1, 1], 100))
    assert repository.count() == 3, "Should count 3 after adding three orders"
    
    # Overwrite an order (same ID)
    repository.add(Order("3", [2, 2], [3, 3], 100))
    assert repository.count() == 3, "Count shouldn't change when overwriting"