from delivery_sim.entities.states import PairState


class PairRepository:
    def __init__(self):
        self.pairs = {}  # Maps pair_id to Pair objects
    
    def add(self, pair):
        self.pairs[pair.pair_id] = pair
    
    def find_by_id(self, pair_id):
        return self.pairs.get(pair_id)
    
    def find_all(self):
        return list(self.pairs.values())
    
    def find_by_state(self, state):
        return [pair for pair in self.pairs.values() if pair.state == state]
    
    def find_unassigned_pairs(self):
        return self.find_by_state(PairState.CREATED)
        
    def find_by_order_id(self, order_id):
        return [pair for pair in self.pairs.values() 
                if pair.order1.order_id == order_id or 
                   pair.order2.order_id == order_id]