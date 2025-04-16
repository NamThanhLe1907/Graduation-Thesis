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
import threading
from queue import Queue


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
            max_depth=1,
            encoder='vits',
            checkpoint_path='utility/checkpoint/depth_anything_v2_metric_hypersim_vits.pth',
            device='cuda'  # Thêm device parameter
        )
        self.running = False

        # Đơn vị hiển thị depth cho bounding box: 'cm' hoặc 'mm'
        self.depth_unit = 'cm'  # hoặc 'mm'

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

    def start_processing(self):

        self.running = True
        self.camera.initialize()

        # Khởi tạo lock và queue cho đồng bộ hóa
        self.frame_lock = threading.Lock()
        self.yolo_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self.frame_queue = Queue(maxsize=1)
        self.yolo_queue = Queue(maxsize=1)
        self.depth_queue = Queue(maxsize=1)

        # Thread YOLO (chạy trước)
        def yolo_loop():
            while self.running:
                try:
                    # Lấy frame mới nhất từ queue
                    frame = self.frame_queue.get_nowait()
                    
                    # Tiền xử lý và chạy YOLO
                    processed = self.processor.preprocess_frame(frame)
                    results = self.yolo_inference.infer(processed)
                    
                    with self.yolo_lock:
                        self.last_yolo_results = results
                        # Clear queue
                        while not self.yolo_queue.empty():
                            self.yolo_queue.get_nowait()
                        self.yolo_queue.put(results)
                        
                        # Nếu có kết quả YOLO, đẩy frame và results vào queue depth
                        if results and len(results) > 0 and results[0].obb:
                            while not self.depth_queue.empty():
                                self.depth_queue.get_nowait()
                            self.depth_queue.put((frame.copy(), results))
                    
                except:
                    pass
                time.sleep(0.001)

        # Thread Depth (chạy sau khi có YOLO)
        def depth_loop():
            frame_counter = 0
            while self.running:
                try:
                    # Lấy frame và kết quả YOLO từ queue
                    try:
                        frame, results = self.depth_queue.get_nowait()
                    except Exception as e_queue:
                        import queue as py_queue
                        if isinstance(e_queue, py_queue.Empty):
                            # Queue rỗng, không phải lỗi, bỏ qua
                            time.sleep(0.005)
                            continue
                        else:
                            # Lỗi khác, in ra
                            import traceback
                            print(f"❌ [DEPTH ERROR] Frame #{frame_counter} | Queue error: {str(e_queue)}")
                            print(f"📜 [STACK TRACE]\n{traceback.format_exc()}")
                            time.sleep(0.01)
                            continue

                    frame_counter += 1
                    
                    if results and len(results) > 0 and results[0].obb:
                        obb = results[0].obb
                        print(f"🔍 [DEPTH] Frame #{frame_counter} | Processing {len(obb)} objects")
                        start_time = time.time()
                        
                        # Chỉ tính depth cho các vùng có object
                        for box in obb.xyxyxyxy:
                            # Crop frame với boundary check
                            if isinstance(box, torch.Tensor):
                                pts = box.detach().cpu().numpy().reshape(4, 2).astype(int)
                            else:
                                pts = np.array(box).reshape(4, 2).astype(int)
                            x, y, w, h = cv2.boundingRect(pts)
                            # Validate boundary
                            x = max(0, x)
                            y = max(0, y)
                            w = min(w, frame.shape[1] - x)
                            h = min(h, frame.shape[0] - y)
                            if w <= 10 or h <= 10:  # Skip quá nhỏ
                                print(f"⚠️ [SKIP] Box too small: {w}x{h}")
                                continue
                            cropped = frame[y:y+h, x:x+w]
                            
                            if cropped.size > 0 and len(cropped.shape) == 3:
                                # Chuyển sang RGB và tính depth
                                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                                if not rgb.flags['C_CONTIGUOUS']:
                                    rgb = np.ascontiguousarray(rgb)
                                
                                input_size = ((max(rgb.shape[:2]) + 13) // 14) * 14
                                depth = self.depth_estimator.model.infer_image(rgb, input_size=input_size)
                                # Dynamic scale based on max_depth
                                depth_scale = self.depth_estimator.max_depth / 10.0  # 10.0 là max_depth mặc định của model
                                depth = depth * depth_scale
                                
                                # Debug và xử lý depth map
                                print(f"🔧 [DEBUG] Depth type: {type(depth)}, shape: {depth.shape if hasattr(depth, 'shape') else 'N/A'}")
                                try:
                                    if isinstance(depth, torch.Tensor):
                                        print(f"⚙️ [TENSOR INFO] Device: {depth.device}, Type: {depth.dtype}")
                                        depth_np = depth.detach().cpu().numpy()
                                        print(f"🔄 [CONVERSION] Converted to numpy array | Shape: {depth_np.shape}")
                                    elif isinstance(depth, np.ndarray):
                                        depth_np = depth
                                        print(f"🔢 [NUMPY ARRAY] Direct use | Shape: {depth_np.shape}")
                                    else:
                                        # Trường hợp không rõ kiểu dữ liệu
                                        print(f"⚠️ [WARNING] Depth data is neither Tensor nor ndarray. Type: {type(depth)}. Attempting to convert to numpy array.")
                                        try:
                                            depth_np = np.array(depth)
                                            print(f"✅ [CONVERTED] Converted to numpy array | Shape: {depth_np.shape}")
                                        except Exception as e_conv:
                                            print(f"❌ [ERROR] Failed to convert depth to numpy array: {e_conv}")
                                            continue  # bỏ qua frame này
                                    
                                    print(f"📊 [DEPTH STATS] Min: {depth_np.min():.2f}, Max: {depth_np.max():.2f}")
                                    
                                    with self.depth_lock:
                                        # Đảm bảo depth_np là numpy array trước khi gọi astype
                                        if not isinstance(depth_np, np.ndarray):
                                            try:
                                                depth_np = np.array(depth_np)
                                            except:
                                                print(f"❌ [ERROR] Cannot convert depth to numpy array for astype(). Skipping this frame.")
                                                continue
                                        self.last_calibrated_depth = depth_np.astype(np.float32)
                                        print(f"🔒 [LOCK ACQUIRED] Depth map saved")
                                        
                                        heatmap, _ = self.depth_estimator.get_heatmap(depth_np, unit='cm')
                                        print(f"🎨 [HEATMAP] Generated | Shape: {heatmap.shape}")
                                        self.last_depth_heatmap = heatmap
                                    
                                except Exception as e:
                                    import traceback
                                    print(f"🔥 [DEPTH ERROR] Frame #{frame_counter} | {str(e)}")
                                    print(f"📜 [STACK TRACE]\n{traceback.format_exc()}")
                                    torch.cuda.empty_cache()
                                    time.sleep(0.1)  # Giảm tải GPU
                                continue
                                
                            torch.cuda.empty_cache()
                        
                        proc_time = time.time() - start_time
                        print(f"✅ [DEPTH] Frame #{frame_counter} | Processed in {proc_time:.2f}s")
                                
                except Exception as e:
                    import traceback
                    print(f"❌ [DEPTH ERROR] Frame #{frame_counter} | {str(e)}")
                    print(f"📜 [STACK TRACE]\n{traceback.format_exc()}")
                time.sleep(0.01)

        threading.Thread(target=yolo_loop, daemon=True).start()
        threading.Thread(target=depth_loop, daemon=True).start()

        while self.running:
            try:
                self.performance_monitor.start_frame()
                self.frame_count += 1

                # Lấy khung hình từ camera
                frame = self.camera.get_frame()
                if frame is None:
                    continue

                # TÍNH FULL FRAME DEPTH NGAY SAU KHI LẤY FRAME
                try:
                    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if not rgb_full.flags['C_CONTIGUOUS']:
                        rgb_full = np.ascontiguousarray(rgb_full)
                    # Đảm bảo kích thước chia hết cho 14
                    input_size = ((max(rgb_full.shape[:2]) + 13) // 14) * 14
                    depth_full = self.depth_estimator.model.infer_image(rgb_full, input_size=input_size)
                    # Scale depth theo max_depth
                    scale_factor = self.depth_estimator.max_depth / 10.0
                    depth_full = depth_full * scale_factor

                    # Convert tensor sang numpy nếu cần
                    if isinstance(depth_full, torch.Tensor):
                        depth_full_np = depth_full.detach().cpu().numpy()
                    elif isinstance(depth_full, np.ndarray):
                        depth_full_np = depth_full
                    else:
                        depth_full_np = np.array(depth_full)

                    self.last_calibrated_depth = depth_full_np.astype(np.float32)
                except:
                    pass
# Tạo heatmap cho full frame depth heatmap_full, _ = self.depth_estimator.get_heatmap(depth_full_np, unit='cm') self.last_depth_heatmap = heatmap_full except Exception as e: print(f"[FULL FRAME DEPTH ERROR] {e}")

                # Đẩy frame vào queue và đảm bảo chỉ giữ frame mới nhất
                try:
                    # Clear queue nếu có nhiều hơn 1 frame chờ xử lý
                    while not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame.copy())
                except:
                    pass

                # Tiền xử lý khung hình
                processed_frame = self.processor.preprocess_frame(frame)
                annotated_frame = processed_frame.copy()
                img_w, img_h = frame.shape[1], frame.shape[0]
                img_size = (img_w, img_h)
                
                # Lấy kết quả YOLO mới nhất
                results = getattr(self, 'last_yolo_results', None)
                  
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
                        
                        from collections import defaultdict
                        class_depths = defaultdict(list)

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
                                        if self.depth_unit == 'mm':
                                            scale_factor = 1000
                                            unit_label = 'mm'
                                            decimals = 0
                                        else:  # mặc định cm
                                            scale_factor = 100
                                            unit_label = 'cm'
                                            decimals = 3
                                        d_min = region.min() * scale_factor
                                        d_mean = region.mean() * scale_factor
                                        d_max = region.max() * scale_factor
                                        label_text += f" | D: {d_mean:.{decimals}f}{unit_label}"
                                        # Lưu mean depth vào dict theo class
                                        class_depths[cls_id].append(d_mean)
                                except Exception as e:
                                    pass
                            
                            xmin = pts[:,0].min()
                            ymax = pts[:,1].max()
                            text_pos = (int(xmin), int(ymax))
                            cv2.putText(annotated_frame, label_text, text_pos,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        
                        # Sau khi duyệt hết các box, tính trung bình theo class
                        # Ghép thông tin khoảng cách trung bình theo class
                        class_depth_info = ""
                        for cls_id, depth_list in class_depths.items():
                            if len(depth_list) > 0:
                                mean_depth_class = np.mean(depth_list)
                                line = f"[Class {cls_id}] Mean distance: {mean_depth_class:.2f} {unit_label} ({len(depth_list)} boxes)"
                                print(line)
                                class_depth_info += line + "\n"

                        # Tính khoảng cách trung bình toàn ảnh (full frame)
                        full_frame_info = ""
                        if self.last_calibrated_depth is not None:
                            try:
                                full_mean = np.mean(self.last_calibrated_depth) * (1000 if self.depth_unit=='mm' else 100)
                                full_min = np.min(self.last_calibrated_depth) * (1000 if self.depth_unit=='mm' else 100)
                                full_max = np.max(self.last_calibrated_depth) * (1000 if self.depth_unit=='mm' else 100)
                                full_frame_info = f"[Full Frame] Mean: {full_mean:.2f} {unit_label}, Min: {full_min:.2f} {unit_label}, Max: {full_max:.2f} {unit_label}"
                                print(full_frame_info)
                            except:
                                pass
                        
                        # Nếu có thông tin full frame, overlay lên annotated_frame
                        if full_frame_info:
                            try:
                                cv2.putText(annotated_frame, full_frame_info, (10, 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            except Exception as e:
                                print(f"Error overlaying full frame info: {e}")

                        # Hiển thị lên GUI console
                        combined_info = ""
                        if full_frame_info:
                            combined_info += full_frame_info + "\n"
                        if class_depth_info.strip():
                            combined_info += class_depth_info.strip()
                        if combined_info.strip():
                            self.gui.log_message(combined_info.strip(), level="INFO")
                        
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
