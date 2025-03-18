import cv2
from utils.camera_interface import CameraInterface
from utils.frame_processor import FrameProcessor
from utils.yolo_inference import YOLOInference
from utils.postprocessing import PostProcessor
from utils.visualization import Visualizer
from utils.performance_monitor import PerformanceMonitor

def main():
    # Khởi tạo các module
    camera = CameraInterface()
    processor = FrameProcessor()
    yolo_inference = YOLOInference()
    post_processor = PostProcessor()
    visualizer = Visualizer()
    performance_monitor = PerformanceMonitor()

    # Khởi động camera
    camera.initialize()

    try:
        while True:
            performance_monitor.start_frame()
            frame = camera.get_frame()

            # Tiền xử lý khung hình
            processed_frame = processor.preprocess_frame(frame)

            # Thực hiện inference
            results = yolo_inference.infer(processed_frame)

            # Hậu xử lý
            boxes = results.xyxy[0][:, :4].cpu().numpy()  # Bounding boxes
            confidences = results.xyxy[0][:, 4].cpu().numpy()  # Confidence scores
            class_ids = results.xyxy[0][:, 5].cpu().numpy().astype(int)  # Class IDs

            smoothed_boxes = post_processor.smooth_boxes(boxes)
            collisions = post_processor.detect_collisions(smoothed_boxes)

            # Hiển thị kết quả
            frame_with_boxes = visualizer.draw_boxes(frame, smoothed_boxes, confidences, class_ids, ["pallet", "load"])
            visualizer.show_frame(frame_with_boxes)

            # Kết thúc frame
            performance_monitor.end_frame()

            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()