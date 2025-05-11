class SequentialIdGenerator:
    """
    Generates sequential numeric IDs.
    
    This simple generator provides increasing integer IDs starting from
    a specified value. Each entity type can have its own generator instance
    to maintain separate ID sequences.
    """
    
    def __init__(self, start=1):
        """
        Initialize the generator with a starting value.
        
        Args:
            start: The first ID to generate (default: 1)
        """
        self.counter = start
    
    def next(self):
        """
        Generate the next ID in sequence.
        
        Returns:
            int: The next sequential ID
        """
        current = self.counter
        self.counter += 1
        return str(current)  # Convert to string