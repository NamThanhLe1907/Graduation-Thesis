import cv2
import numpy as np
import torch
import threading
import time
import logging
from queue import Queue, Empty, Full
from collections import defaultdict

from utility import (CameraInterface,
                    QueueManager,
                    FrameProcessor,
                    YOLOInference,
                    PostProcessor,
                    PerformanceMonitor,
                    DepthEstimatorV2)


class FramePublisher:
    def __init__(self, camera):
        self.camera = camera
        self.frame_queue = Queue(maxsize=1)
        self.lock = threading.Lock()
        self.running = False
        self.frame_counter = 0  # Thêm frame counter

    def start(self):
        self.running = True
        threading.Thread(target=self._frame_loop, daemon=True).start()

    def _frame_loop(self):
        while self.running:
            try:
                frame = self.camera.get_frame()
                with self.lock:
                    # Clear queue if full to keep only latest frame
                    while not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                    self.frame_counter += 1
                    self.frame_queue.put_nowait((frame, self.frame_counter))
                    # print(f"FramePublisher: Put frame into queue with shape {frame.shape}")
            except Exception as e:
                print(f"FramePublisher error: {e}")
            time.sleep(0.001)

    def get_latest_frame(self):
        with self.lock:
            try:
                return self.frame_queue.get_nowait()
            except:
                return None, None

    def stop(self):
        self.running = False

class YOLOProcessor:
    def __init__(self, frame_publisher, gui, depth_ready_event, frame_to_depth_queue):
        self.frame_publisher = frame_publisher
        self.gui = gui
        self.processor = FrameProcessor()
        self.yolo_inference = YOLOInference(
            model_path="final.pt",
            conf=0.9
        )
        self.post_processor = PostProcessor(alpha=0.2)
        self.running = False
        self.depth_ready_event = depth_ready_event
        self.frame_to_depth_queue = frame_to_depth_queue
        self.performance_monitor = PerformanceMonitor()
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def start(self):
        self.running = True
        threading.Thread(target=self._yolo_loop, daemon=True).start()

    def _yolo_loop(self):
        while self.running:
            # Wait for depth processing to be ready
            self.depth_ready_event.wait()
            frame, frame_id = self.frame_publisher.get_latest_frame()
            if frame is None:
                time.sleep(0.01)  # Increase sleep to reduce load
                continue
            try:
                processed = self.processor.preprocess_frame(frame)
                results = self.yolo_inference.infer(processed)
                start_time = time.time()
                self.logger.info(f"YOLOProcessor: Processing frame id {frame_id}")
                # Push frame to depth queue for synchronized depth processing
                try:
                    if not self.frame_to_depth_queue.full():
                        self.frame_to_depth_queue.put(frame.copy(), timeout=0.1)
                        put_time = time.time() - start_time
                        self.performance_monitor.record_latency('queue_put', put_time)
                except Full:
                    self.logger.warning("Frame queue full, dropping frame")
                except Exception as e:
                    self.logger.error(f"Error putting frame to depth queue: {e}")
            except Exception as e:
                self.logger.error(f"YOLO inference error: {e}")
                results = None

            if results and len(results) > 0 and results[0].obb:
                annotated_frame = processed.copy()
                self._process_yolo_results(results, annotated_frame, processed, (frame.shape[1], frame.shape[0]))
                self.gui.root.after(0, lambda: self.gui.update(annotated_frame))
            else:
                self.logger.debug("No inference results")
            time.sleep(0.02)  # Increase sleep to reduce load

    def _process_yolo_results(self, results, annotated_frame, processed_frame, img_size):
        obb = results[0].obb
        confs = obb.conf.detach().cpu().numpy()
        self.logger.info(f"Confidence stats - Min: {confs.min():.2f}, Max: {confs.max():.2f}, Mean: {confs.mean():.2f}")

        valid_conf_indices = np.where(confs > 0.9)[0]
        if valid_conf_indices.size == 0:
            return

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

        self.logger.info(f"Geometry filter: Kept {len(boxes)}/{original_count} boxes")

        collisions = self.post_processor.detect_collisions(
            obb,
            frame_resolution=img_size,
            confs=confs,
            valid_indices=valid_indices
        )

        rotated_boxes = obb.xyxyxyxy.detach().cpu().numpy()
        indices = valid_indices if valid_indices is not None else range(len(rotated_boxes))

        for idx in indices:
            poly = rotated_boxes[idx]
            cls_id = obb.cls.detach().cpu().numpy()[idx]
            conf = confs[idx]
            color = (255, 255, 0)  # Default color

            if hasattr(results[0], "names"):
                names = results[0].names
            else:
                names = {0: "-load-", 1: "-pallet-"}

            label = names.get(cls_id, f"ID:{cls_id}")
            angle_deg = float(np.rad2deg(obb.xywhr[idx, 4].cpu().numpy())) if idx < len(obb.xywhr) else None

            label_text = f"{label} {conf:.2f}" + (f", {angle_deg:.2f}deg" if angle_deg else "")

            pts = poly.reshape(4, 2).astype(int)
            cv2.polylines(annotated_frame, [pts], isClosed=True, color=color, thickness=4)

            xmin = pts[:, 0].min()
            ymax = pts[:, 1].max()
            text_pos = (int(xmin), int(ymax))
            cv2.putText(annotated_frame, label_text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


class DepthProcessor:
    def __init__(self, frame_publisher, gui, depth_ready_event):
        self.frame_publisher = frame_publisher
        self.gui = gui
        self.depth_ready_event = depth_ready_event
        self.depth_estimator = DepthEstimatorV2(
            max_depth=1,
            encoder='vits',
            checkpoint_path='utility/checkpoint/depth_anything_v2_metric_hypersim_vits.pth',
            device='cuda'
        )
        # Cấu hình tối ưu
        self.frame_skip = 20 # Xử lý 1/3 frame
        self.downsample_ratio = 0.5  # Giảm kích thước ảnh 75%
        self.running = False
        self.depth_unit = 'cm'
        self.last_depth_heatmap = None
        self.last_calibrated_depth = None
        self.frame_count = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def start(self):
        self.running = True
        threading.Thread(target=self._depth_loop, daemon=True).start()

    def _depth_loop(self):
        self.logger.info("DepthProcessor thread started")
        while self.running:
            try:
                # Clear event to indicate depth processing started
                self.depth_ready_event.clear()
                frame, frame_id = self.frame_publisher.get_latest_frame()
                if frame is None:
                    time.sleep(0.005)
                    self.depth_ready_event.set()
                    continue
                self.frame_count += 1
                self.logger.info(f"DepthProcessor received frame id {frame_id} with shape: {frame.shape if frame is not None else 'None'}")
                if self.frame_count % self.frame_skip == 0:
                    self.logger.info(f"DepthProcessor processing frame id {frame_id}")
                    # Giảm kích thước ảnh trước khi xử lý
                    small_frame = cv2.resize(frame, (0,0), fx=self.downsample_ratio, fy=self.downsample_ratio)
                    rgb_full = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    if not rgb_full.flags['C_CONTIGUOUS']:
                        rgb_full = np.ascontiguousarray(rgb_full)
                    input_size = ((max(rgb_full.shape[:2]) + 13) // 14) * 14
                    depth_full = self.depth_estimator.model.infer_image(rgb_full, input_size=input_size)
                    scale_factor = self.depth_estimator.max_depth / 10.0
                    depth_full = depth_full * scale_factor

                    if isinstance(depth_full, torch.Tensor):
                        depth_full_np = depth_full.detach().cpu().numpy()
                    elif isinstance(depth_full, np.ndarray):
                        depth_full_np = depth_full
                    else:
                        depth_full_np = np.array(depth_full)

                    self.last_calibrated_depth = depth_full_np.astype(np.float32)

                    heatmap_full, _ = self.depth_estimator.get_heatmap(depth_full_np, unit=self.depth_unit)

                    self.logger.info(f"Heatmap shape: {heatmap_full.shape}, dtype: {heatmap_full.dtype}, min: {heatmap_full.min()}, max: {heatmap_full.max()}")

                    if len(heatmap_full.shape) == 2:
                        heatmap_bgr = cv2.cvtColor(heatmap_full, cv2.COLOR_GRAY2BGR)
                    else:
                        heatmap_bgr = heatmap_full

                    self.last_depth_heatmap = heatmap_bgr

                    self.logger.info("Updating GUI with new depth heatmap")
                    self.gui.root.after(0, lambda: self.gui.update_depth(self.last_depth_heatmap))
                # Set event to indicate depth processing finished
                self.depth_ready_event.set()
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"DepthProcessor loop error: {e}")
                time.sleep(0.1)


class VideoProcessorV2:
    def __init__(self, gui):
        self.gui = gui
        self.performance_monitor = PerformanceMonitor()
        self.camera = CameraInterface()
        self.frame_publisher = FramePublisher(self.camera)
        self.depth_ready_event = threading.Event()
        self.depth_ready_event.set()
        self.frame_to_depth_queue = QueueManager(maxsize=1)
        self.yolo_processor = YOLOProcessor(
            self.frame_publisher,
            self.gui,
            self.depth_ready_event,
            self.frame_to_depth_queue
        )
        self.depth_processor = DepthProcessor(self.frame_publisher, self.gui, self.depth_ready_event)
        self.running = False

    def start_processing(self):
        self.running = True
        self.camera.initialize()
        self.frame_publisher.start()
        self.yolo_processor.start()
        self.depth_processor.start()

    def stop_processing(self):
        self.running = False
        self.frame_publisher.stop()
        self.yolo_processor.running = False
        self.depth_processor.running = False
