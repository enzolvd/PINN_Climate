import json
import os
from pathlib import Path
import argparse
from typing import List, Dict

class ExperimentQueue:
    def __init__(self, queue_dir: str = "experiment_queue"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(exist_ok=True)
        self.queue_file = self.queue_dir / "queue.json"
        self.current_file = self.queue_dir / "current.json"
        self.completed_file = self.queue_dir / "completed.json"
        
        # Initialize queue files if they don't exist
        if not self.queue_file.exists():
            self._save_json(self.queue_file, [])
        if not self.completed_file.exists():
            self._save_json(self.completed_file, [])
            
    def _load_json(self, file_path: Path) -> List[Dict]:
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return []

    def _save_json(self, file_path: Path, data: List[Dict]):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_experiments(self, experiments: List[Dict]):
        """Add new experiments to the queue."""
        current_queue = self._load_json(self.queue_file)
        current_queue.extend(experiments)
        self._save_json(self.queue_file, current_queue)
        print(f"Added {len(experiments)} experiments to queue")

    def get_next_experiment(self) -> Dict:
        """Get the next experiment from the queue and mark it as current."""
        queue = self._load_json(self.queue_file)
        if not queue:
            return None
        
        # Get next experiment and remove it from queue
        next_exp = queue.pop(0)
        self._save_json(self.queue_file, queue)
        
        # Save as current experiment
        self._save_json(self.current_file, next_exp)
        
        return next_exp

    def mark_current_completed(self):
        """Mark the current experiment as completed and remove current."""
        if self.current_file.exists():
            current = self._load_json(self.current_file)
            completed = self._load_json(self.completed_file)
            completed.append(current)
            self._save_json(self.completed_file, completed)
            self.current_file.unlink()  # Delete current file

    def get_queue_status(self) -> Dict:
        """Get the current status of the queue."""
        queue = self._load_json(self.queue_file)
        completed = self._load_json(self.completed_file)
        current = self._load_json(self.current_file) if self.current_file.exists() else None
        
        return {
            "queued": len(queue),
            "completed": len(completed),
            "current": current,
            "queue": queue,
        }

def main():
    parser = argparse.ArgumentParser(description="Manage experiment queue")
    parser.add_argument("--action", choices=["add", "status", "get_next", "mark_completed"], required=True)
    parser.add_argument("--experiments", type=str, help="JSON file containing experiments to add")
    args = parser.parse_args()

    queue = ExperimentQueue()

    if args.action == "add" and args.experiments:
        with open(args.experiments, 'r') as f:
            experiments = json.load(f)
        queue.add_experiments(experiments)
    
    elif args.action == "status":
        status = queue.get_queue_status()
        print(json.dumps(status, indent=2))
    
    elif args.action == "get_next":
        next_exp = queue.get_next_experiment()
        if next_exp:
            print(json.dumps(next_exp))
    
    elif args.action == "mark_completed":
        queue.mark_current_completed()

if __name__ == "__main__":
    main()