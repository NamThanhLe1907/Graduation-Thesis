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
                if results and len(results) > 0:
                    try:
                        # Debug: in ra thông tin của result[0]
                        # print("Attributes of result[0]:", dir(results[0]))
                        # print("Result[0] content:", results[0].__dict__)

                        # Trích xuất bounding boxes: ưu tiên dùng result[0].boxes nếu có, nếu không thì dùng result[0].obb
                        if results[0].boxes is not None:
                            boxes_obj = results[0].boxes
                            print("Using result[0].boxes")
                        elif hasattr(results[0], "obb") and results[0].obb is not None:
                            boxes_obj = results[0].obb
                            print("Using result[0].obb")
                        else:
                            boxes_obj = None

                        if boxes_obj is None:
                            print("No bounding boxes found in YOLO result.")
                        else:
                            # Với YOLOv8 OBB, ưu tiên dùng thuộc tính xyxyxyxy để lấy box dạng 8 tọa độ
                            if hasattr(boxes_obj, "xyxyxyxy"):
                                boxes = boxes_obj.xyxyxyxy
                                if not isinstance(boxes, np.ndarray):
                                    boxes = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.array(boxes)
                                # Chuyển từ dạng (N, 4, 2) về (N, 8)
                                boxes = boxes.reshape(boxes.shape[0], -1)
                                # print("Extracted boxes from xyxyxyxy:", boxes)
                            elif hasattr(boxes_obj, "xyxy"):
                                boxes = boxes_obj.xyxy
                                if not isinstance(boxes, np.ndarray):
                                    boxes = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.array(boxes)
                                # print("Extracted boxes from xyxy:", boxes)
                            else:
                                boxes = boxes_obj  # Giả sử nó đã là numpy array
                                # print("Extracted boxes directly:", boxes)

                            # Lấy thông tin classification: class id và confidence (nếu có)
                            # Giả sử boxes_obj có thuộc tính cls và conf
                            if hasattr(boxes_obj, "cls"):
                                cls_tensor = boxes_obj.cls
                                if not isinstance(cls_tensor, np.ndarray):
                                    cls_ids = cls_tensor.cpu().numpy().astype(int) if hasattr(cls_tensor, "cpu") else np.array(cls_tensor).astype(int)
                                else:
                                    cls_ids = cls_tensor.astype(int)
                                print("Extracted class ids:", cls_ids)
                            else:
                                cls_ids = np.zeros(boxes.shape[0], dtype=int)

                            if hasattr(boxes_obj, "conf"):
                                conf_tensor = boxes_obj.conf
                                if not isinstance(conf_tensor, np.ndarray):
                                    confs = conf_tensor.cpu().numpy() if hasattr(conf_tensor, "cpu") else np.array(conf_tensor)
                                else:
                                    confs = conf_tensor
                                print("Extracted confidences:", confs)
                            else:
                                confs = np.ones(boxes.shape[0])

                            if hasattr(boxes_obj, "xywhr"):
                                xywhr = boxes_obj.xywhr
                                if not isinstance(xywhr, np.ndarray):
                                    xywhr = xywhr.cpu().numpy() if hasattr(xywhr, "cpu") else np.array(xywhr)
                                print("Extracted xywhr:", xywhr)
                            else:
                                xywhr = None    
                                
                            # Lấy dictionary mapping class id sang tên
                            names = results[0].names if hasattr(results[0], "names") else {}
                            print("Names mapping:", names)

                            if boxes is None or boxes.size == 0:
                                print("No bounding boxes data available after extraction.")
                            else:
                                # --- QUAN TRỌNG ---
                                # Nếu box nhảy loạn, có thể do smoothing gây ra.
                                # Hãy tắt smoothing nếu cần.
                                use_smoothing = False
                                if use_smoothing:
                                    smoothed_boxes = self.post_processor.smooth_boxes(boxes)
                                else:
                                    smoothed_boxes = boxes

                                # Chú ý: hàm detect_collisions chỉ được thiết kế cho box axis-aligned (4 tọa độ).
                                # Với OBB (8 tọa độ), thuật toán này có thể cần được điều chỉnh.
                                collisions = self.post_processor.detect_collisions(smoothed_boxes)

                                # Vẽ các bounding box và nhãn
                                annotated_frame = processed_frame.copy()
                                for i, box in enumerate(smoothed_boxes):
                                    # Chọn màu dựa trên class id, nếu không có thì dùng màu mặc định
                                    cls_id = cls_ids[i] if i < len(cls_ids) else 0
                                    color = self.color_map.get(cls_id, (255, 255, 0))  # Mặc định: vàng
                                    label = names.get(cls_id, f"ID:{cls_id}")
                                    confidence = confs[i] if i < len(confs) else 1.0
                                    
                                    if xywhr is not None and i < xywhr.shape[0]:
                                        angle_rad = xywhr[i, 4]
                                        angle_deg = angle_rad * 180.0 / math.pi
                                    else:
                                        angle_deg = None

                                    # Xây dựng label text hiển thị tên, confidence và góc xoay
                                    if angle_deg is not None:
                                        label_text = f"{label} {confidence:.2f}, {angle_deg:.1f}°"
                                    else:
                                        label_text = f"{label} {confidence:.2f}"
                                    
                                    if len(box) == 4:
                                        x1, y1, x2, y2 = map(int, box)
                                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                        # Đặt nhãn tại vị trí 3/4 của chiều cao từ trên xuống
                                        text_y = y1 + int(0.75 * (y2 - y1))
                                        cv2.putText(annotated_frame, label_text, (x1, text_y), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    # Nếu box OBB (8 tọa độ)
                                    elif len(box) == 8:
                                        pts = box.reshape((-1, 1, 2)).astype(np.int32)
                                        cv2.polylines(annotated_frame, [pts], isClosed=True, color=color, thickness=2)
                                        # Tính bounding rect của polygon để đặt nhãn
                                        x, y, w, h = cv2.boundingRect(pts)
                                        text_y = y + int(0.75 * h)
                                        cv2.putText(annotated_frame, label_text, (x, text_y), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                                # Vẽ chỉ báo va chạm (nếu cần)
                                for (i, j) in collisions:
                                    box1 = smoothed_boxes[i]
                                    box2 = smoothed_boxes[j]
                                    if len(box1) == 4:
                                        center1 = (int((box1[0] + box1[2]) / 2), int((box1[1] + box1[3]) / 2))
                                    elif len(box1) == 8:
                                        pts1 = box1.reshape(-1, 2)
                                        center1 = tuple(np.mean(pts1, axis=0).astype(int))
                                    if len(box2) == 4:
                                        center2 = (int((box2[0] + box2[2]) / 2), int((box2[1] + box2[3]) / 2))
                                    elif len(box2) == 8:
                                        pts2 = box2.reshape(-1, 2)
                                        center2 = tuple(np.mean(pts2, axis=0).astype(int))
                                    cv2.circle(annotated_frame, center1, 5, (0, 0, 255), -1)
                                    cv2.circle(annotated_frame, center2, 5, (0, 0, 255), -1)
                                    cv2.line(annotated_frame, center1, center2, (0, 0, 255), 2)

                                diff = cv2.norm(annotated_frame, processed_frame)
                                self.gui.log_message(f"Annotation applied with postprocessing, diff={diff}, shape={annotated_frame.shape}", "DEBUG")
                    except Exception as e:
                        print(f"Error in postprocessing: {e}")
                        annotated_frame = processed_frame.copy()
                else:
                    print("No inference results")

                # Lưu file debug để kiểm tra kết quả
                cv2.imwrite("debug_output.jpg", annotated_frame)

                # Chuyển đổi màu nếu cần (ví dụ từ BGRA sang BGR)
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGRA2BGR) if (len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 4) else annotated_frame
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
