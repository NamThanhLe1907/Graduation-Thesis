import threading
from queue import Queue

class QueueManager:
    def __init__(self, maxsize=1):
        self.queue = Queue(maxsize=maxsize)
        self.lock = threading.Lock()
        
    def put(self, item, timeout=0.1):
        with self.lock:
            # Clear queue if full
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except:
                    break
            try:
                self.queue.put(item, timeout=timeout)
                return True
            except:
                return False

    def full(self):
        """Check if queue is full"""
        with self.lock:
            return self.queue.full()

    def get(self, timeout=0.1):
        with self.lock:
            try:
                return self.queue.get(timeout=timeout)
            except:
                return None

    def clear(self):
        with self.lock:
            while not self.queue.empty():
                self.queue.get_nowait()
