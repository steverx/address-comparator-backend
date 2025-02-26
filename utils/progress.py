from typing import Dict
import threading
import logging

logger = logging.getLogger(__name__)

class ProgressTracker:
    def __init__(self):
        self._progress = {}
        self._lock = threading.Lock()

    def update_progress(self, job_id: str, progress: Dict):
        with self._lock:
            self._progress[job_id] = progress
            logger.info(f"Job {job_id} progress: {progress}")

    def get_progress(self, job_id: str) -> Dict:
        with self._lock:
            return self._progress.get(job_id, {})

progress_tracker = ProgressTracker()