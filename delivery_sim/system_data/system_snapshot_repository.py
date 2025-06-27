class SystemSnapshotRepository:
    def __init__(self):
        self.snapshots = []  # List of snapshot dictionaries
    
    def add_snapshot(self, snapshot_dict):
        self.snapshots.append(snapshot_dict)
    
    def get_snapshots_in_range(self, start_time=None, end_time=None):
        filtered = self.snapshots
        if start_time is not None:
            filtered = [s for s in filtered if s['timestamp'] >= start_time]
        if end_time is not None:
            filtered = [s for s in filtered if s['timestamp'] <= end_time]
        return filtered
    
    def get_all_snapshots(self):
        return self.snapshots.copy()