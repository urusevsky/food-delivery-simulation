from delivery_sim.utils.logging_system import get_logger

class SystemDataCollector:
    """
    Orchestrates system snapshot collection during simulation.
    
    Handles timing, collection intervals, and storage coordination.
    Delegates actual metric calculations to SystemDataDefinitions.
    """
    
    def __init__(self, env, system_data_definitions, snapshot_repository, collection_interval=1.0):
        """
        Initialize system data collector.
        
        Args:
            env: SimPy environment
            system_data_definitions: SystemDataDefinitions instance for calculations
            snapshot_repository: SystemSnapshotRepository for storage
            collection_interval: Minutes between snapshots
        """
        self.env = env
        self.definitions = system_data_definitions
        self.snapshot_repository = snapshot_repository
        self.collection_interval = collection_interval
        self.logger = get_logger("system_data.collector")
        
        self.logger.info(f"[t={self.env.now:.2f}] SystemDataCollector initialized "
                        f"with interval={collection_interval} minutes")
        self.process = env.process(self._collection_process())
    
    def _collection_process(self):
        """SimPy process that collects snapshots at regular intervals."""
        while True:
            # Get snapshot data from definitions
            snapshot_data = self.definitions.create_snapshot_data(self.env.now)
            
            # Store in repository
            self.snapshot_repository.add_snapshot(snapshot_data)
            
            self.logger.debug(f"[t={self.env.now:.2f}] Collected snapshot: "
                            f"available_drivers={snapshot_data['available_drivers']}")
            
            # Wait for next collection interval
            yield self.env.timeout(self.collection_interval)