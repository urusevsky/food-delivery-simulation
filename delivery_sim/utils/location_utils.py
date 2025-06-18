def calculate_distance(loc1, loc2):
    """
    Calculate Euclidean distance between two locations.
    
    Args:
        loc1: First location [x, y]
        loc2: Second location [x, y]
        
    Returns:
        float: Euclidean distance between locations
    """
    return ((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)**0.5

def locations_are_equal(loc1, loc2):
    """
    Check if two locations are the same.
    
    Args:
        loc1: First location [x, y]
        loc2: Second location [x, y]
        
    Returns:
        bool: True if locations are equal, False otherwise
    """
    return loc1[0] == loc2[0] and loc1[1] == loc2[1]

def format_location(location, precision=2):
    """
    Format location coordinates for readable logging.
    
    Args:
        location: Location as [x, y] list or tuple
        precision: Number of decimal places (default: 2)
        
    Returns:
        str: Formatted location string like "[3.45, 7.89]"
    """
    return f"[{location[0]:.{precision}f}, {location[1]:.{precision}f}]"