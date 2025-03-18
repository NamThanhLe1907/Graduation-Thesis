import torch
import cv2

class YOLOInference:
    def __init__(self, model_path='best.pt', device='cuda'):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path, force_reload=True)
        self.model.to(self.device)

    def infer(self, frame):
        """Run inference on a given frame."""
        results = self.model(frame)
        return results

if __name__ == "__main__":
    yolo_inference = YOLOInference()
    # Example usage
    frame = cv2.imread('sample_image.jpg')  # Replace with actual frame capture
    results = yolo_inference.infer(frame)
    results.show()  # Display results