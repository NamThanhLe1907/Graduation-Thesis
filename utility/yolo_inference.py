from ultralytics import YOLO
import cv2

class YOLOInference:
    def __init__(self, model_path="best.pt", conf=None, iou=0.5):
        self.model = YOLO(model_path, task='obb').to('cpu')
        # Set detection thresholds
        self.model.conf = conf  # Confidence threshold
        self.model.iou = iou    # NMS IoU threshold

    def infer(self, frame):
        """Run inference on a given frame."""
        # For OBB models we need to specify the task and use imgsz from the model
        results = self.model(frame, task='obb', imgsz=self.model.args['imgsz'], device='cpu')
        return results

if __name__ == "__main__":
    inference = YOLOInference(model_path="best.pt",conf=0.1)
    frame = cv2.imread("./utils/sample_image.jpg")  # Replace with actual frame capture
    if frame is None:
        print("Error: Could not read 'sample_image.jpg'.")
    else:
        results = inference.infer(frame)
        # Use plot() method from ultralytics to annotate detections
        annotated_frame = results[0].plot()
        cv2.imshow("Inference Results", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
