import cv2
import numpy as np

class FrameProcessor:
    def __init__(self):
        pass

    def adjust_contrast(self, frame):
        """Adjust contrast dynamically."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

    def reduce_noise(self, frame):
        """Reduce noise using Non-local Means Denoising."""
        return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    def preprocess_frame(self, frame):
        """Preprocess the frame: adjust contrast and reduce noise."""
        frame = self.adjust_contrast(frame)
        frame = self.reduce_noise(frame)
        return frame

if __name__ == "__main__":
    processor = FrameProcessor()
    # Example usage
    frame = cv2.imread('sample_image.jpg')  # Replace with actual frame capture
    processed_frame = processor.preprocess_frame(frame)
    cv2.imshow("Processed Frame", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()