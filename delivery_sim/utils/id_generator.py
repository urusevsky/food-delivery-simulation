class PrefixedIdGenerator:
    """
    Generates sequential IDs with a consistent prefix.
    
    This generator creates IDs that include a type prefix, making it
    immediately clear what kind of entity an ID refers to. The separator
    between prefix and number can be customized.
    
    Examples:
        - With separator='': O1, O2, O3 (compact format)
        - With separator='-': ORD-1, ORD-2 (readable format)
    """
    
    def __init__(self, prefix, start=1, separator=''):
        """
        Initialize the generator with prefix and formatting options.
        
        Args:
            prefix: The prefix to add to all IDs (e.g., 'O' for orders)
            start: The first number to use (default: 1)
            separator: String to place between prefix and number (default: '')
        """
        self.prefix = prefix
        self.counter = start
        self.separator = separator
    
    def next(self):
        """
        Generate the next prefixed ID in sequence.
        
        Returns:
            str: The next ID with prefix (e.g., 'O1', 'D2')
        """
        current = self.counter
        self.counter += 1
        return f"{self.prefix}{self.separator}{current}"