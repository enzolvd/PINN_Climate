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
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if not content:  # Empty file
                        return []
                    return json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON from {file_path}. Using empty list instead.")
                return []
        return []

    def _save_json(self, file_path: Path, data: List[Dict]):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_experiments(self, experiments: List[Dict]):
        """Add new experiments to the queue, avoiding duplicates."""
        # Load current state
        current_queue = self._load_json(self.queue_file)
        completed = self._load_json(self.completed_file)
        current = self._load_json(self.current_file) if self.current_file.exists() else []
        
        # Get existing experiment names
        existing_names = set()
        
        # Add names from queue
        existing_names.update(exp["experiment_name"] for exp in current_queue)
        
        # Add names from completed
        existing_names.update(exp["experiment_name"] for exp in completed)
        
        # Add current if exists
        if isinstance(current, dict):
            existing_names.add(current["experiment_name"])
        elif isinstance(current, list) and current:
            existing_names.update(exp["experiment_name"] for exp in current)
        
        # Filter out experiments that already exist
        new_experiments = [
            exp for exp in experiments 
            if exp["experiment_name"] not in existing_names
        ]
        
        if new_experiments:
            current_queue.extend(new_experiments)
            self._save_json(self.queue_file, current_queue)
            print(f"Added {len(new_experiments)} new experiments to queue")
        else:
            print("No new experiments to add - all already exist in queue/current/completed")

    def get_next_experiment(self) -> Dict:
        """Get the next experiment from the queue."""
        # First check if there's a current experiment
        if self.current_file.exists():
            current = self._load_json(self.current_file)
            return current
            
        # If no current experiment, get next from queue
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
        
        # Handle current file more carefully
        current = None
        if self.current_file.exists():
            try:
                current = self._load_json(self.current_file)
                # If we got an empty list but expected a dict, set to None
                if isinstance(current, list) and not current:
                    current = None
            except Exception as e:
                print(f"Error loading current file: {e}")
        
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
    parser.add_argument("--queue_dir", type=str, default="experiment_queue", 
                        help="Directory to store queue files (default: experiment_queue)")
    args = parser.parse_args()

    queue = ExperimentQueue(queue_dir=args.queue_dir)

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
            # Only print the JSON data, no additional messages
            print(json.dumps(next_exp))
    
    elif args.action == "mark_completed":
        queue.mark_current_completed()

if __name__ == "__main__":
    main()