import cv2
import numpy as np
import torch
from utility.camera_interface import CameraInterface
from utility.frame_processor import FrameProcessor
from utility.yolo_inference import YOLOInference
from utility.postprocessing import PostProcessor
from utility.performance_monitor import PerformanceMonitor
# from src.Module_division import DivisionModule
import time
from utility.depth_calculate_v2 import DepthEstimatorV2

class VideoProcessor:
    def __init__(self, gui):
        self.gui = gui
        self.camera = CameraInterface()
        self.processor = FrameProcessor()
        self.yolo_inference = YOLOInference(
            model_path="final.pt", 
            conf=0.9
        )
        # self.division_module = DivisionModule(pallet_config={'LR': 12, 'LQ': 10})
        self.post_processor = PostProcessor(alpha=0.2)
        self.performance_monitor = PerformanceMonitor()
        self.depth_estimator = DepthEstimatorV2(
            max_depth=5.75,
            encoder='vitb',
            checkpoint_path='utility/checkpoint/depth_anything_v2_metric_hypersim_vitb.pth',
            device='cuda'  # Thêm device parameter
        )
        self.running = False

        # Định nghĩa bảng màu cho các lớp (bạn có thể điều chỉnh theo số lượng và mong muốn)
        self.color_map = {
            0: (255, 0, 0),   # Blue
            1: (0, 255, 0),   # Green
            2: (0, 0, 255),   # Red
            # Thêm các màu khác nếu cần
        }
        # Biến lưu heatmap và depth map giảm tải inference
        self.last_depth_heatmap = None
        self.last_calibrated_depth = None
        self.frame_count = 0

    def _update_depth_estimation(self, frames):
        try:
            # Xử lý batch frames
            batch_tensor = torch.stack([
                torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1)
                for frame in frames
            ]).to(self.depth_estimator.device) / 255.0
            
            with torch.inference_mode(), torch.cuda.amp.autocast():
                depths = self.depth_estimator.model.infer(batch_tensor)
                torch.cuda.empty_cache()
            
            # Lấy frame cuối cùng để hiển thị
            calibrated_depth = depths[-1].cpu().numpy().astype(np.float32)
            depth_heatmap, _ = self.depth_estimator.get_heatmap(calibrated_depth)
            self.last_depth_heatmap = depth_heatmap
            self.last_calibrated_depth = calibrated_depth
            
            # Lưu debug image mỗi 10 frame
            if self.frame_count % 10 == 0:
                cv2.imwrite("debug_output.jpg", cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Depth estimation error: {e}")

    def start_processing(self):
        self.running = True
        self.camera.initialize()

        while self.running:
            try:
                self.performance_monitor.start_frame()
                self.frame_count += 1

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
                
                # Xử lý kết quả YOLO (như trước)
                if results and len(results) > 0:
                    if results[0].obb:
                        obb = results[0].obb
                        confs = obb.conf.detach().cpu().numpy()
                        print(f"Confidence stats - Min: {confs.min():.2f}, Max: {confs.max():.2f}, Mean: {confs.mean():.2f}")
                        valid_conf_indices = np.where(confs > 0.9)[0]
                        if valid_conf_indices.size == 0:
                            continue
                        confs = confs[valid_conf_indices]
                        boxes_all = obb.xywhr.detach().cpu().numpy()
                        boxes = boxes_all[valid_conf_indices]
                        original_count = len(boxes)
                        filtered_result = self.post_processor.filter_by_geometry(
                            boxes,
                            frame_resolution=img_size
                        )
                        if isinstance(filtered_result, tuple):
                            boxes, valid_indices = filtered_result
                            confs = confs[valid_indices]
                        else:
                            boxes = filtered_result
                            valid_indices = None
                        print(f"Geometry filter: Kept {len(boxes)}/{original_count} boxes")
                            
                        collisions = self.post_processor.detect_collisions(
                            obb,
                            frame_resolution=img_size,
                            confs=confs,
                            valid_indices=valid_indices
                        )
                        
                        annotated_frame = processed_frame.copy()
                        rotated_boxes = obb.xyxyxyxy.detach().cpu().numpy()
                        indices = valid_indices if valid_indices is not None else range(len(rotated_boxes))
                        
                        for idx in indices:
                            poly = rotated_boxes[idx]
                            cls_id = obb.cls.detach().cpu().numpy()[idx]
                            conf = confs[idx]
                            color = self.color_map.get(cls_id, (255, 255, 0))
                            
                            if hasattr(results[0], "names"):
                                names = results[0].names
                            else:
                                names = {0: "-load-", 1: "-pallet-"}
                            
                            label = names.get(cls_id, f"ID:{cls_id}")
                            angle_deg = float(np.rad2deg(obb.xywhr[idx, 4])) if idx < len(obb.xywhr) else None
                            
                            label_text = f"{label} {conf:.2f}" + (f", {angle_deg:.2f}deg" if angle_deg else "")
                            
                            pts = poly.reshape(4, 2).astype(int)
                            cv2.polylines(annotated_frame, [pts], isClosed=True, color=color, thickness=4)
                            
                            # Tính toán và hiển thị depth cho từng box
                            if self.last_calibrated_depth is not None:
                                try:
                                    x, y, w, h = cv2.boundingRect(pts)
                                    # Giới hạn vùng không vượt quá kích thước depth map
                                    x_end = min(x + w, self.last_calibrated_depth.shape[1])
                                    y_end = min(y + h, self.last_calibrated_depth.shape[0])
                                    region = self.last_calibrated_depth[y:y_end, x:x_end]
                                    if region.size > 0:
                                        d_min = region.min()
                                        d_mean = region.mean()
                                        d_max = region.max()
                                        label_text += f" | D: {d_mean:.2f}m"
                                except Exception as e:
                                    pass
                            
                            xmin = pts[:,0].min()
                            ymax = pts[:,1].max()
                            text_pos = (int(xmin), int(ymax))
                            cv2.putText(annotated_frame, label_text, text_pos,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                            
                        for (i, j) in collisions:
                            box1 = boxes[i]
                            box2 = boxes[j]
                            x_center1 = int(box1[0])
                            y_center1 = int(box1[1])
                            center1 = (x_center1, y_center1)
                            
                            x_center2 = int(box2[0])
                            y_center2 = int(box2[1])
                            center2 = (x_center2, y_center2)
                            cv2.circle(annotated_frame, center1, 5, (0, 0, 255), -1)
                            cv2.circle(annotated_frame, center2, 5, (0, 0, 255), -1)
                            cv2.line(annotated_frame, center1, center2, (0, 0, 255), 2)
                            
                        diff = cv2.norm(annotated_frame, processed_frame)
                        self.gui.log_message(f"Annotation applied with postprocessing, diff={diff}, shape={annotated_frame.shape}", "DEBUG")
                else:
                    print("No inference results")
                
                # Tích hợp batch processing mỗi 5 frame
                if self.frame_count % 5 == 0:
                    # Thu thập 5 frame liên tiếp để xử lý batch
                    batch_frames = [self.camera.get_frame() for _ in range(5)]
                    self._update_depth_estimation(batch_frames)
                # Nếu đã có heatmap từ vòng trước, sử dụng lại
                # Đảm bảo annotated_frame là numpy array hợp lệ
                if annotated_frame is not None and isinstance(annotated_frame, np.ndarray):
                    if self.last_depth_heatmap is not None and isinstance(self.last_depth_heatmap, np.ndarray):
                        if annotated_frame.shape == self.last_depth_heatmap.shape:
                            try:
                                combined_frame = cv2.addWeighted(annotated_frame, 0.7, self.last_depth_heatmap, 0.3, 0)
                            except Exception as e:
                                print(f"Error combining frames: {str(e)}")
                                combined_frame = annotated_frame.copy()
                        else:
                            combined_frame = annotated_frame.copy()
                    else:
                        combined_frame = annotated_frame.copy()
                    
                    # Lưu file debug nếu cần
                    try:
                        cv2.imwrite("debug_output.jpg", combined_frame)
                    except Exception as e:
                        print(f"Error saving debug image: {str(e)}")
                else:
                    combined_frame = np.zeros((480, 640, 3), dtype=np.uint8)

                if combined_frame is not None and len(combined_frame.shape) == 3:
                    if combined_frame.shape[2] == 4:
                        display_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGRA2BGR)
                    else:
                        display_frame = combined_frame.copy()
                else:
                    display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                self.gui.root.after(0, lambda: self.gui.update(display_frame))
                if self.last_depth_heatmap is not None:
                    self.gui.root.after(0, lambda: self.gui.update_depth(self.last_depth_heatmap))

                self.performance_monitor.end_frame()
                fps = self.performance_monitor.get_fps()
                avg_inference = self.performance_monitor.get_average_inference_time()
                self.gui.update_fps_info(fps, avg_inference)
                time.sleep(0.01)

            except Exception as e:
                print(f"Processing error: {str(e)}")
                self.gui.log_message(f"Processing error: {str(e)}", "ERROR")
                break

        self.camera.release()
        self.gui.log_message("Camera released", "INFO")
        print("Camera released")
