def log_entity_not_found(service_name, entity_type, entity_id, timestamp=None):
    """
    Centralized logging for entity not found errors.
    
    Args:
        service_name: Name of the service reporting the error
        entity_type: Type of entity that wasn't found
        entity_id: ID of the entity that wasn't found
        timestamp: Optional current simulation time
    """
    time_info = f" at time {timestamp}" if timestamp is not None else ""
    print(f"Error: {entity_type} {entity_id} not found in {service_name}{time_info}")