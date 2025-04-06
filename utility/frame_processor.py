import cv2
import numpy as np

class FrameProcessor:
    def __init__(self):
        pass

    def preprocess_frame(self, frame):
        """Preprocess the frame: adjust contrast and reduce noise."""

        return frame

if __name__ == "__main__":
    processor = FrameProcessor()
    # Example usage
    frame = cv2.imread('sample_image.jpg')  # Replace with actual frame capture
    processed_frame = processor.preprocess_frame(frame)
    cv2.imshow("Processed Frame", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()