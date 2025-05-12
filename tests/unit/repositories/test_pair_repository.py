
import pytest
from delivery_sim.repositories.pair_repository import PairRepository
from delivery_sim.entities.pair import Pair
from delivery_sim.entities.order import Order
from delivery_sim.entities.states import PairState

# Test Group 1: Basic repository operations
def test_repository_initialization():
    """
    Test that a new pair repository starts empty with proper initialization.
    This ensures our repository has a clean starting state for storing pairs.
    """
    # ARRANGE & ACT
    repository = PairRepository()
    
    # ASSERT
    assert hasattr(repository, 'pairs'), "Repository should have a pairs dictionary"
    assert isinstance(repository.pairs, dict), "Pairs storage should be a dictionary"
    assert len(repository.pairs) == 0, "New repository should start empty"
    assert repository.find_all() == [], "find_all should return empty list"

def test_add_pair():
    """
    Test that pairs can be added to the repository.
    This verifies the fundamental storage operation works correctly.
    """
    # ARRANGE
    repository = PairRepository()
    order1 = Order("101", [0, 0], [1, 1], 100)
    order2 = Order("102", [0, 0], [2, 2], 105)
    pair = Pair(order1, order2, 110)
    
    # ACT
    repository.add(pair)
    
    # ASSERT
    assert pair.pair_id in repository.pairs, "Pair ID should be in the repository"
    assert repository.pairs[pair.pair_id] is pair, "Repository should store the exact pair object"
    assert len(repository.pairs) == 1, "Repository should contain one pair"

def test_add_multiple_pairs():
    """
    Test that multiple pairs can be added and stored correctly.
    This verifies the repository handles multiple entities properly.
    """
    # ARRANGE
    repository = PairRepository()
    
    # Create first pair
    order1_1 = Order("101", [0, 0], [1, 1], 100)
    order1_2 = Order("102", [0, 0], [2, 2], 105)
    pair1 = Pair(order1_1, order1_2, 110)
    
    # Create second pair
    order2_1 = Order("103", [1, 1], [2, 2], 110)
    order2_2 = Order("104", [1, 1], [3, 3], 115)
    pair2 = Pair(order2_1, order2_2, 120)
    
    # ACT
    repository.add(pair1)
    repository.add(pair2)
    
    # ASSERT
    assert len(repository.pairs) == 2, "Repository should contain two pairs"
    assert all(pair_id in repository.pairs for pair_id in [pair1.pair_id, pair2.pair_id])

# Test Group 2: Finding pairs
def test_find_by_id_existing_pair():
    """
    Test that an existing pair can be found by its ID.
    This is essential for retrieving specific pairs during simulation.
    """
    # ARRANGE
    repository = PairRepository()
    order1 = Order("101", [0, 0], [1, 1], 100)
    order2 = Order("102", [0, 0], [2, 2], 105)
    pair = Pair(order1, order2, 110)
    repository.add(pair)
    
    # ACT
    found_pair = repository.find_by_id(pair.pair_id)
    
    # ASSERT
    assert found_pair is pair, "Should return the exact pair object"
    assert found_pair.pair_id == pair.pair_id, "Found pair should have correct ID"

def test_find_by_id_nonexistent_pair():
    """
    Test that find_by_id returns None for nonexistent pairs.
    This handles the case where a pair ID doesn't exist.
    """
    # ARRANGE
    repository = PairRepository()
    order1 = Order("101", [0, 0], [1, 1], 100)
    order2 = Order("102", [0, 0], [2, 2], 105)
    pair = Pair(order1, order2, 110)
    repository.add(pair)
    
    # ACT
    found_pair = repository.find_by_id("999-999")  # Nonexistent ID
    
    # ASSERT
    assert found_pair is None, "Should return None for nonexistent pair"

def test_find_all():
    """
    Test that find_all returns all pairs in the repository.
    This is useful for global operations or statistics.
    """
    # ARRANGE
    repository = PairRepository()
    
    # Create two pairs
    order1_1 = Order("101", [0, 0], [1, 1], 100)
    order1_2 = Order("102", [0, 0], [2, 2], 105)
    pair1 = Pair(order1_1, order1_2, 110)
    
    order2_1 = Order("103", [1, 1], [2, 2], 110)
    order2_2 = Order("104", [1, 1], [3, 3], 115)
    pair2 = Pair(order2_1, order2_2, 120)
    
    repository.add(pair1)
    repository.add(pair2)
    
    # ACT
    all_pairs = repository.find_all()
    
    # ASSERT
    assert len(all_pairs) == 2, "Should return all pairs"
    assert pair1 in all_pairs, "Should include first pair"
    assert pair2 in all_pairs, "Should include second pair"
    assert isinstance(all_pairs, list), "Should return a list"

# Test Group 3: Finding by state
def test_find_by_state():
    """
    Test that pairs can be filtered by their state.
    This is crucial for finding pairs at specific stages of processing.
    """
    # ARRANGE
    repository = PairRepository()
    
    # Create pairs in different states
    order1_1 = Order("101", [0, 0], [1, 1], 100)
    order1_2 = Order("102", [0, 0], [2, 2], 105)
    pair1 = Pair(order1_1, order1_2, 110)  # CREATED by default
    
    order2_1 = Order("103", [1, 1], [2, 2], 110)
    order2_2 = Order("104", [1, 1], [3, 3], 115)
    pair2 = Pair(order2_1, order2_2, 120)
    
    order3_1 = Order("105", [2, 2], [3, 3], 120)
    order3_2 = Order("106", [2, 2], [4, 4], 125)
    pair3 = Pair(order3_1, order3_2, 130)
    
    # Transition pairs to different states
    pair2.state = PairState.ASSIGNED
    pair3.state = PairState.COMPLETED
    
    repository.add(pair1)
    repository.add(pair2)
    repository.add(pair3)
    
    # ACT
    created_pairs = repository.find_by_state(PairState.CREATED)
    assigned_pairs = repository.find_by_state(PairState.ASSIGNED)
    completed_pairs = repository.find_by_state(PairState.COMPLETED)
    
    # ASSERT
    assert len(created_pairs) == 1, "Should find one CREATED pair"
    assert pair1 in created_pairs, "Should find the correct CREATED pair"
    
    assert len(assigned_pairs) == 1, "Should find one ASSIGNED pair"
    assert pair2 in assigned_pairs, "Should find the correct ASSIGNED pair"
    
    assert len(completed_pairs) == 1, "Should find one COMPLETED pair"
    assert pair3 in completed_pairs, "Should find the correct COMPLETED pair"

# Test Group 4: Specialized queries
def test_find_unassigned_pairs():
    """
    Test the specialized method for finding unassigned pairs.
    This is the companion to find_unassigned_orders and together they provide
    all entities that need driver assignment.
    """
    # ARRANGE
    repository = PairRepository()
    
    # Create pairs in various states
    order1_1 = Order("101", [0, 0], [1, 1], 100)
    order1_2 = Order("102", [0, 0], [2, 2], 105)
    unassigned_pair1 = Pair(order1_1, order1_2, 110)  # CREATED (unassigned)
    
    order2_1 = Order("103", [1, 1], [2, 2], 110)
    order2_2 = Order("104", [1, 1], [3, 3], 115)
    unassigned_pair2 = Pair(order2_1, order2_2, 120)  # CREATED (unassigned)
    
    order3_1 = Order("105", [2, 2], [3, 3], 120)
    order3_2 = Order("106", [2, 2], [4, 4], 125)
    assigned_pair = Pair(order3_1, order3_2, 130)
    assigned_pair.state = PairState.ASSIGNED  # Already assigned
    
    repository.add(unassigned_pair1)
    repository.add(unassigned_pair2)
    repository.add(assigned_pair)
    
    # ACT
    unassigned_pairs = repository.find_unassigned_pairs()
    
    # ASSERT
    assert len(unassigned_pairs) == 2, "Should find two unassigned pairs"
    assert unassigned_pair1 in unassigned_pairs, "Should include first unassigned pair"
    assert unassigned_pair2 in unassigned_pairs, "Should include second unassigned pair"
    assert assigned_pair not in unassigned_pairs, "Should not include assigned pair"

def test_find_by_order_id():
    """
    Test finding pairs that contain a specific order.
    This is useful when we need to find the pair containing a particular order.
    """
    # ARRANGE
    repository = PairRepository()
    
    # Create multiple pairs
    order1 = Order("101", [0, 0], [1, 1], 100)
    order2 = Order("102", [0, 0], [2, 2], 105)
    pair1 = Pair(order1, order2, 110)
    
    order3 = Order("103", [1, 1], [2, 2], 110)
    order4 = Order("104", [1, 1], [3, 3], 115)
    pair2 = Pair(order3, order4, 120)
    
    order5 = Order("105", [2, 2], [3, 3], 120)
    order6 = Order("106", [2, 2], [4, 4], 125)
    pair3 = Pair(order5, order6, 130)
    
    repository.add(pair1)
    repository.add(pair2)
    repository.add(pair3)
    
    # ACT
    # Find pairs containing order "101"
    pairs_with_101 = repository.find_by_order_id("101")
    
    # Find pairs containing order "104"
    pairs_with_104 = repository.find_by_order_id("104")
    
    # Find pairs containing non-existent order
    pairs_with_999 = repository.find_by_order_id("999")
    
    # ASSERT
    assert len(pairs_with_101) == 1, "Should find one pair containing order 101"
    assert pair1 in pairs_with_101, "Should find the correct pair"
    
    assert len(pairs_with_104) == 1, "Should find one pair containing order 104"
    assert pair2 in pairs_with_104, "Should find the correct pair"
    
    assert len(pairs_with_999) == 0, "Should find no pairs for non-existent order"

# Test Group 5: Edge cases
def test_duplicate_pair_ids():
    """
    Test what happens when trying to add a pair with duplicate ID.
    This verifies that the repository handles ID conflicts.
    """
    # ARRANGE
    repository = PairRepository()
    
    # Create first pair
    order1_1 = Order("101", [0, 0], [1, 1], 100)
    order1_2 = Order("102", [0, 0], [2, 2], 105)
    pair1 = Pair(order1_1, order1_2, 110)
    
    # Create another pair that happens to have the same ID
    # (This could happen if orders are paired in different order)
    order2_1 = Order("101", [1, 1], [2, 2], 120)
    order2_2 = Order("102", [1, 1], [3, 3], 125)
    pair2 = Pair(order2_1, order2_2, 130)
    
    # ACT
    repository.add(pair1)
    repository.add(pair2)  # This will overwrite pair1
    
    # ASSERT
    assert len(repository.pairs) == 1, "Should still have only one pair"
    found_pair = repository.find_by_id(pair1.pair_id)
    assert found_pair is pair2, "Later pair should overwrite earlier one"
    assert found_pair.creation_time == 130, "Should have pair2's data"

def test_repository_isolation():
    """
    Test that find_all returns a new list, not the internal storage.
    This ensures external code can't accidentally modify the repository.
    """
    # ARRANGE
    repository = PairRepository()
    order1 = Order("101", [0, 0], [1, 1], 100)
    order2 = Order("102", [0, 0], [2, 2], 105)
    pair = Pair(order1, order2, 110)
    repository.add(pair)
    
    # ACT
    all_pairs = repository.find_all()
    all_pairs.clear()  # Try to clear the returned list
    
    # ASSERT
    assert len(repository.pairs) == 1, "Repository should still contain the pair"
    assert len(repository.find_all()) == 1, "Pair should still be in repository"