import cv2
import tkinter as tk
import numpy as np
from threading import Thread
from utils.camera_interface import CameraInterface
from utils.frame_processor import FrameProcessor
from utils.yolo_inference import YOLOInference
from utils.postprocessing import PostProcessor
from utils.visualization import AppGUI
from utils.performance_monitor import PerformanceMonitor

import math
import time

class VideoProcessor:
    def __init__(self, gui):
        self.gui = gui
        self.camera = CameraInterface()
        self.processor = FrameProcessor()
        self.yolo_inference = YOLOInference(model_path="best.pt")  # Hãy chắc chắn model path đúng
        self.post_processor = PostProcessor(alpha=0.2)
        self.performance_monitor = PerformanceMonitor()
        self.running = False

        # Định nghĩa bảng màu cho các lớp (bạn có thể điều chỉnh theo số lượng và mong muốn)
        self.color_map = {
            0: (255, 0, 0),   # Blue
            1: (0, 255, 0),   # Green
            2: (0, 0, 255),   # Red
            # Thêm các màu khác nếu cần
        }

    def start_processing(self):
        self.running = True
        self.camera.initialize()

        while self.running:
            try:
                self.performance_monitor.start_frame()

                # Lấy khung hình từ camera
                frame = self.camera.get_frame()
                if frame is None:
                    continue

                # Tiền xử lý khung hình
                processed_frame = self.processor.preprocess_frame(frame)
                annotated_frame = processed_frame.copy()

                # Chạy inference của YOLO
                results = self.yolo_inference.infer(processed_frame)
                print(f"YOLO results: {len(results)} detections")
                
                # Log confidence và tỷ lệ W/H
                if results and len(results) > 0:
                    boxes_obj = results[0].boxes if results[0].boxes is not None else results[0].obb
                    if boxes_obj:
                        # Lấy thông tin confidence
                        confs = boxes_obj.conf.cpu().numpy() if hasattr(boxes_obj.conf, "cpu") else boxes_obj.conf
                        print(f"Confidence stats - Min: {confs.min():.2f}, Max: {confs.max():.2f}, Mean: {confs.mean():.2f}")
                        
                        # Lấy thông tin kích thước box
                        if hasattr(boxes_obj, "xywhr"):
                            boxes = boxes_obj.xywhr.cpu().numpy() if hasattr(boxes_obj.xywhr, "cpu") else boxes_obj.xywhr
                            # Thêm rotation=0 nếu cần để đảm bảo dữ liệu có 5 cột
                            boxes = np.hstack([boxes, np.zeros((len(boxes), 1))]) if len(boxes) > 0 else boxes
                            print("Using xywhr for boxes")
                        else:
                            boxes = boxes_obj.xywhn.cpu().numpy() if hasattr(boxes_obj.xywhn, "cpu") else boxes_obj.xywhn
                        
                        # Tùy chọn smoothing (mặc định tắt)
                        use_smoothing = False
                        if use_smoothing:
                            smoothed_boxes = self.post_processor.smooth_boxes(boxes)
                        else:
                            smoothed_boxes = boxes
                        
                        # Áp dụng bộ lọc hình học
                        original_count = len(smoothed_boxes)
                        smoothed_boxes = self.post_processor.filter_by_geometry(
                            smoothed_boxes,
                            img_size=(frame.shape[1], frame.shape[0])
                        )
                        print(f"Geometry filter: Kept {len(smoothed_boxes)}/{original_count} boxes")
                        
                        # Phát hiện va chạm giữa các box
                        collisions = self.post_processor.detect_collisions(
                            smoothed_boxes, 
                            img_size=(frame.shape[1], frame.shape[0])
                        )
                        
                        # Vẽ các bounding box và nhãn
                        annotated_frame = processed_frame.copy()
                        for i, box in enumerate(smoothed_boxes):
                            # Xác định class id nếu có thông tin
                            cls_id = 0
                            if hasattr(boxes_obj, "cls"):
                                cls_tensor = boxes_obj.cls
                                cls_ids = cls_tensor.cpu().numpy().astype(int) if hasattr(cls_tensor, "cpu") else np.array(cls_tensor).astype(int)
                                if i < len(cls_ids):
                                    cls_id = cls_ids[i]
                            color = self.color_map.get(cls_id, (255, 255, 0))  # Mặc định: vàng
                            
                            # Lấy confidence của box
                            confidence = confs[i] if i < len(confs) else 1.0
                            
                            # Lấy tên nhãn nếu có
                            names = results[0].names if hasattr(results[0], "names") else {}
                            label = names.get(cls_id, f"ID:{cls_id}")
                            
                            # Lấy góc xoay nếu sử dụng xywhr
                            angle_deg = None
                            if hasattr(boxes_obj, "xywhr") and boxes_obj.xywhr is not None:
                                xywhr = boxes_obj.xywhr.cpu().numpy() if hasattr(boxes_obj.xywhr, "cpu") else boxes_obj.xywhr
                                if i < xywhr.shape[0]:
                                    angle_rad = xywhr[i, 4]
                                    angle_deg = angle_rad * 180.0 / math.pi
                            
                            if angle_deg is not None:
                                label_text = f"{label} {confidence:.2f}, {angle_deg:.1f}°"
                            else:
                                label_text = f"{label} {confidence:.2f}"
                            
                            # Convert OBB to XYXY using postprocessor
                            img_size = (frame.shape[1], frame.shape[0])
                            xyxy_box = self.post_processor.convert_to_xyxy(np.array([box]), img_size)[0]
                            
                            x1, y1, x2, y2 = map(int, xyxy_box)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            text_y = y1 + int(0.75 * (y2 - y1))
                            cv2.putText(annotated_frame, label_text, (x1, text_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Vẽ chỉ báo va chạm
                        for (i, j) in collisions:
                            box1 = smoothed_boxes[i]
                            box2 = smoothed_boxes[j]
                            # Get center from xywhr format (normalized coordinates)
                            img_w, img_h = img_size
                            x_center = int(box1[0] * img_w)
                            y_center = int(box1[1] * img_h)
                            center1 = (x_center, y_center)
                            
                            x_center = int(box2[0] * img_w)
                            y_center = int(box2[1] * img_h)
                            center2 = (x_center, y_center)
                            cv2.circle(annotated_frame, center1, 5, (0, 0, 255), -1)
                            cv2.circle(annotated_frame, center2, 5, (0, 0, 255), -1)
                            cv2.line(annotated_frame, center1, center2, (0, 0, 255), 2)
                            
                        diff = cv2.norm(annotated_frame, processed_frame)
                        self.gui.log_message(f"Annotation applied with postprocessing, diff={diff}, shape={annotated_frame.shape}", "DEBUG")
                else:
                    print("No inference results")
                
                # Lưu file debug để kiểm tra kết quả
                cv2.imwrite("debug_output.jpg", annotated_frame)

                # Chuyển đổi màu nếu cần (ví dụ, từ BGRA sang BGR)
                if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 4:
                    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGRA2BGR)
                else:
                    display_frame = annotated_frame
                self.gui.root.after(0, lambda: self.gui.update(display_frame))

                self.performance_monitor.end_frame()
                time.sleep(0.01)

            except Exception as e:
                print(f"Processing error: {str(e)}")
                self.gui.log_message(f"Processing error: {str(e)}", "ERROR")
                break

        self.camera.release()
        self.gui.log_message("Camera released", "INFO")
        print("Camera released")

def main():
    root = tk.Tk()
    gui = AppGUI(root)
    processor = VideoProcessor(gui)

    processing_thread = Thread(target=processor.start_processing)
    processing_thread.daemon = True
    processing_thread.start()

    print("Ứng dụng đang khởi động...")
    gui.run()
    print("Ứng dụng đã tắt")

if __name__ == "__main__":
    print("Bắt đầu chạy chương trình từ main")
    main()
