import numpy as np

class PostProcessor:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.previous_boxes = None

    def smooth_boxes(self, current_boxes):
        """Smooth bounding boxes using Exponential Moving Average."""
        if self.previous_boxes is None:
            self.previous_boxes = current_boxes
            return current_boxes
        
        smoothed_boxes = []
        for current, previous in zip(current_boxes, self.previous_boxes):
            smoothed_box = (self.alpha * current + (1 - self.alpha) * previous)
            smoothed_boxes.append(smoothed_box)
        
        self.previous_boxes = smoothed_boxes
        return smoothed_boxes

    def detect_collisions(self, boxes):
        """Detect collisions between bounding boxes."""
        collisions = []
        for i, box1 in enumerate(boxes):
            for j, box2 in enumerate(boxes):
                if i != j and self._boxes_intersect(box1, box2):
                    collisions.append((i, j))
        return collisions

    def _boxes_intersect(self, box1, box2):
        """Check if two boxes intersect."""
        return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

if __name__ == "__main__":
    processor = PostProcessor()
    # Example usage
    current_boxes = np.array([[100, 100, 200, 200], [150, 150, 250, 250]])  # Example bounding boxes
    smoothed_boxes = processor.smooth_boxes(current_boxes)
    print("Smoothed Boxes:", smoothed_boxes)
    collisions = processor.detect_collisions(current_boxes)
    print("Collisions Detected:", collisions)