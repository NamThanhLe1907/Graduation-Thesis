import time

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.frames_processed = 0
        self.total_inference_time = 0

    def start_frame(self):
        """Start timing for a new frame."""
        self.frame_start_time = time.time()

    def end_frame(self):
        """End timing for the frame and update metrics."""
        self.frames_processed += 1
        self.total_inference_time += (time.time() - self.frame_start_time)

    def get_fps(self):
        """Calculate frames per second (FPS)."""
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return 0
        return self.frames_processed / elapsed_time

    def get_average_inference_time(self):
        """Calculate average inference time per frame."""
        if self.frames_processed == 0:
            return 0
        return self.total_inference_time / self.frames_processed

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    # Example usage
    for _ in range(5):  # Simulate processing 5 frames
        monitor.start_frame()
        time.sleep(0.1)  # Simulate inference time
        monitor.end_frame()
    print("FPS:", monitor.get_fps())
    print("Average Inference Time:", monitor.get_average_inference_time())