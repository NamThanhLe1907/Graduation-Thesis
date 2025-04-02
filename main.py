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
                img_w, img_h = frame.shape[1], frame.shape[0]
                img_size = (img_w, img_h)
                
                # Chạy inference của YOLO
                results = self.yolo_inference.infer(processed_frame)
                print(f"YOLO results: {len(results)} detections")
                
                # Log confidence và tỷ lệ W/H
                if results and len(results) > 0:
                    if results[0].obb:
                        obb = results[0].obb
                        # Lấy thông tin confidence
                        confs = obb.conf.cpu().numpy() if hasattr(obb.conf, "cpu") else obb.conf
                        print(f"Confidence stats - Min: {confs.min():.2f}, Max: {confs.max():.2f}, Mean: {confs.mean():.2f}")
                        
                        # Lấy thông tin kích thước box từ OBB
                        boxes = obb.xywhr.cpu().numpy() if hasattr(obb.xywhr, "cpu") else obb.xywhr
                        
                        # Tùy chọn smoothing (mặc định tắt)
                        use_smoothing = False
                        if use_smoothing:
                            smoothed_boxes = self.post_processor.smooth_boxes(boxes)
                        else:
                            smoothed_boxes = boxes
                        
                        # Áp dụng bộ lọc hình học
                        original_count = len(smoothed_boxes)
                        filtered_result = self.post_processor.filter_by_geometry(
                            smoothed_boxes,
                            img_size=img_size
                        )
                        if isinstance(filtered_result, tuple):
                            smoothed_boxes, valid_indices = filtered_result
                        else:
                            smoothed_boxes = filtered_result
                            valid_indices = None
                        print(f"Geometry filter: Kept {len(smoothed_boxes)}/{original_count} boxes")
                        
                        # Phát hiện va chạm giữa các box
                        collisions = self.post_processor.detect_collisions(
                            obb,  # Truyền trực tiếp OBB object
                            img_size=img_size,
                            valid_indices=valid_indices
                        )
                        
                        # Vẽ các bounding box và nhãn
                        annotated_frame = processed_frame.copy()
                        # Lấy tọa độ các điểm polygon xoay
                        # Lấy tọa độ các điểm polygon xoay và chỉ số hợp lệ
                        rotated_boxes = obb.xyxyxyxy.cpu().numpy() if hasattr(obb.xyxyxyxy, "cpu") else obb.xyxyxyxy
                        indices = valid_indices if valid_indices is not None else range(len(rotated_boxes))
                        
                        for idx in indices:
                            poly = rotated_boxes[idx]
                            cls_id = obb.cls.cpu().numpy()[idx]
                            conf = confs[idx]
                            color = self.color_map.get(cls_id, (255, 255, 0))  # Mặc định: vàng
                            
                            # Lấy tên nhãn và góc xoay
                            names = results[0].names if hasattr(results[0], "names") else {}
                            label = names.get(cls_id, f"ID:{cls_id}")
                            angle_deg = np.rad2deg(obb.xywhr[idx, 4]) if idx < len(obb.xywhr) else None
                            
                            label_text = f"{label} {conf:.2f}" + (f", {angle_deg:.1f}°" if angle_deg else "")
                            
                            # Vẽ polygon xoay
                            pts = poly.reshape(4, 2).astype(int)
                            cv2.polylines(annotated_frame, [pts], isClosed=True, color=color, thickness=2)
                            
                            # Tính toán vị trí text dựa trên tâm box
                            xc = int(obb.xywhr[idx][0] * img_w)
                            yc = int(obb.xywhr[idx][1] * img_h)
                            text_pos = (xc - 50, yc - 10)  # Canh giữa và lùi trái 50px
                            cv2.putText(annotated_frame, label_text, text_pos,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(annotated_frame, label_text, text_pos,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Vẽ chỉ báo va chạm
                        for (i, j) in collisions:
                            box1 = smoothed_boxes[i]
                            box2 = smoothed_boxes[j]
                            # Get center from xywhr format (normalized coordinates)
                            x_center1 = int(box1[0] * img_w)
                            y_center1 = int(box1[1] * img_h)
                            center1 = (x_center1, y_center1)
                            
                            x_center2 = int(box2[0] * img_w)
                            y_center2 = int(box2[1] * img_h)
                            center2 = (x_center2, y_center2)
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
