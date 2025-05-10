import unittest
import multiprocessing as mp 
import time 
import numpy as np 
import sys 
import os 
import cv2

# Add folder in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from  utility import (CameraInterface,
                      YOLOInference,
                      DepthEstimator)

# Cấu hình test - đặt USE_CAMERA = False để sử dụng ảnh tĩnh thay vì camera
USE_CAMERA = False
IMAGE_PATH = "tests/assets/image_3.jpg"  # Đường dẫn tới ảnh test nếu USE_CAMERA = False
CONF_THRESH = 0.25  # Ngưỡng confidence cho YOLO

def camera_process(frame_queue, run_event, ready_event):
    """Process camera frames in a separate process"""
    try:
        print(f"[Camera P{os.getpid()}] Starting...")
        
        if not USE_CAMERA:
            # Chế độ ảnh tĩnh
            image_path = find_image(IMAGE_PATH)
            if not image_path:
                print(f"[Camera P{os.getpid()}] Error: Test image not found!")
                return
                
            print(f"[Camera P{os.getpid()}] Loading image from: {image_path}")
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"[Camera P{os.getpid()}] Error: Failed to read image")
                return
                
            print(f"[Camera P{os.getpid()}] Image loaded successfully, shape: {frame.shape}")
            
            # Báo hiệu đã sẵn sàng
            ready_event.set()
            print(f"[Camera P{os.getpid()}] Ready (Image mode)")
            
            # Gửi cùng một ảnh nhiều lần để giả lập camera
            count = 0  
            while run_event.is_set() and count < 20:  # Giới hạn số lần gửi
                frame_queue.put(frame.copy())  # Gửi bản sao của frame
                count += 1
                if count % 5 == 0:
                    print(f"[Camera P{os.getpid()}] Sent image {count} times")
                time.sleep(0.5)  # Đợi 0.5 giây giữa các lần gửi
        else:
            # Chế độ camera thật
            camera = CameraInterface()
            camera.initialize()
            
            # Báo hiệu đã sẵn sàng
            ready_event.set()
            print(f"[Camera P{os.getpid()}] Ready (Camera mode)")
            
            count = 0  
            while run_event.is_set():
                try:
                    frame = camera.get_frame()
                    frame_queue.put(frame)
                    count += 1
                    if count % 10 == 0:
                        print(f"[Camera P{os.getpid()}] Processed {count} frames")
                    time.sleep(0.01)
                except Exception as e:
                    print(f"[Camera P{os.getpid()}] Error: {e}")
                    time.sleep(0.1)
    except Exception as e:
        print(f"[Camera P{os.getpid()}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[Camera P{os.getpid()}] Exiting...")
        if USE_CAMERA and 'camera' in locals():
            camera.release()

def find_image(image_name):
    """Tìm ảnh ở nhiều vị trí khác nhau"""
    # Đường dẫn tuyệt đối
    image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), image_name)
    if os.path.exists(image_path):
        return image_path
        
    # Thử tìm ảnh ở vị trí khác
    alt_paths = [
        f"./{image_name}",
        f"../{image_name}",
        f"tests/{image_name}",
        f"utility/{image_name}",
    ]
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            return os.path.abspath(alt_path)
            
    return None  # Không tìm thấy

def yolo_process(frame_queue, detection_queue, depth_info_queue, run_event, ready_event):
    """Process YOLO inference in a separate process"""
    try:
        print(f"[YOLO P{os.getpid()}] Starting...")
        # Sử dụng đường dẫn tuyệt đối đến model
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utility/best.pt")
        print(f"[YOLO P{os.getpid()}] Loading model from: {model_path}")
        yolo = YOLOInference(model_path=model_path, conf=CONF_THRESH)
        
        ready_event.set()
        print(f"[YOLO P{os.getpid()}] Ready")
        
        count = 0 
        while run_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.5)
                if frame is not None:
                    results = yolo.infer(frame)
                    bounding_boxes = []
                    
                    # Thử lấy từ OBB trước
                    if hasattr(results[0], 'obb') and len(results[0].obb) > 0:
                        for i in range(len(results[0].obb)):
                            box = results[0].obb.xyxy[i].cpu().numpy()
                            bounding_boxes.append({
                                'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                            })
                    
                    # Fallback nếu không có obb - dùng boxes thông thường
                    if len(bounding_boxes) == 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                        for i in range(len(results[0].boxes)):
                            box = results[0].boxes.xyxy[i].cpu().numpy()
                            bounding_boxes.append({
                                'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                            })
                            
                    detections = {
                        'bounding_boxes': bounding_boxes,
                        'frame_number': count+1
                    }
                    
                    detection_queue.put((frame, detections))
                    
                    depth_info_queue.put({
                        'frame': frame,
                        'bounding_boxes': bounding_boxes
                    })

                    count += 1
                    if USE_CAMERA:
                        if count % 10 == 0:
                            print(f"[YOLO P{os.getpid()}] has processed {count} detections, found {len(bounding_boxes)} objects")
                    else:
                        print(f"[YOLO P{os.getpid()}] Processed detection #{count}, found {len(bounding_boxes)} objects")
            except Exception as e:
                if isinstance(e, mp.queues.Empty):
                    # Queue rỗng, bỏ qua không in lỗi
                    time.sleep(0.2)  # Đợi lâu hơn khi queue rỗng
                    continue  # Bỏ qua phần code bên dưới
                else:
                    print(f"[YOLO P{os.getpid()}] Error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
    except Exception as e:
        print(f"[YOLO P{os.getpid()}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[YOLO P{os.getpid()}] Exiting...")

def depth_process(depth_info_queue, depth_results_queue, run_event, ready_event):
    """Process depth estimation in a separate process"""
    try:
        print(f"[Depth P{os.getpid()}] Starting...")
        depth_model = DepthEstimator()
        
        ready_event.set()
        print(f"[Depth P{os.getpid()}] Ready")
        
        count = 0 
        while run_event.is_set():
            try:
                depth_info = depth_info_queue.get(timeout=0.5)
                if depth_info is not None:
                    frame = depth_info['frame']
                    bounding_boxes = depth_info['bounding_boxes']
                    
                    if len(bounding_boxes) > 0:
                        print(f"[Depth P{os.getpid()}] Processing {len(bounding_boxes)} bounding boxes")
                        depth_results = depth_model.estimate_depth(frame, bounding_boxes)
                        depth_results_queue.put((frame, depth_results))
                        
                        count += 1
                        print(f"[Depth P{os.getpid()}] Processed depth #{count} for {len(depth_results)} objects")
                    else:
                        print(f"[Depth P{os.getpid()}] No bounding boxes to process")
                        
            except Exception as e:
                if isinstance(e, mp.queues.Empty):
                    # Queue rỗng, bỏ qua không in lỗi
                    time.sleep(0.2)  # Đợi lâu hơn khi queue rỗng
                    continue  # Bỏ qua phần code bên dưới
                else:
                    print(f"[Depth P{os.getpid()}] Error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
    except Exception as e:
        print(f"[Depth P{os.getpid()}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[Depth P{os.getpid()}] Exiting...")
        
class TestFullPipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up the test environmentj"""
        # Make sure multiprocessing is initialized and using spawn method
        if mp.get_start_method() != 'spawn':
            try:
                mp.set_start_method('spawn', force = True)
            except RuntimeError:
                print("Failed to set spawn method as start method")
                
        print(f"Multiprocessing start method: {mp.get_start_method()}")
        
        # Initialize queues
        self.frame_queue = mp.Queue(maxsize = 5)
        self.detection_queue = mp.Queue(maxsize = 5)
        self.depth_info_queue = mp.Queue(maxsize = 5)
        self.depth_results_queue = mp.Queue(maxsize = 5)
        
        # Initialize events
        self.run_event = mp.Event()
        self.ready_event = mp.Event()
        self.camera_ready = mp.Event()
        self.yolo_ready = mp.Event()
        self.depth_ready = mp.Event()
        
        # Set run event
        self.run_event.set()
        
        # Process objects
        self.camera_process = None
        self.yolo_process = None
        self.depth_process = None
    
    def tearDown(self):
        """Tear down the test environment"""
        # Stop all processes
        if self.run_event.is_set():
            print("Stopping all processes...")
            self.run_event.clear()
            
        # Waiting for all processes to finish
        if self.camera_process:
            self.camera_process.join(timeout = 1)
            print(f"Camera Process has been stopped: {not self.camera_process.is_alive()}")
            
            
        if self.yolo_process:
            self.yolo_process.join(timeout = 1)
            print(f"YOLO Process has been stopped: {not self.yolo_process.is_alive()}")
            
        if self.depth_process:
            self.depth_process.join(timeout = 1)
            print(f"Depth Process has been stopped: {not self.depth_process.is_alive()}")

        print("All processes have been stopped")
        
    def test_full_pipeline(self):
        """Test the full pipeline"""
        
        print(f"Running test with {'CAMERA' if USE_CAMERA else 'STATIC IMAGE'} mode")
        print("Initializing processes...")
        
        self.camera_process = mp.Process(
            target = camera_process,
            args = (self.frame_queue,
                    self.run_event,
                    self.camera_ready),
            daemon = True
        )

        self.yolo_process = mp.Process(
            target = yolo_process,
            args = (self.frame_queue, 
                    self.detection_queue,
                    self.depth_info_queue,
                    self.run_event,
                    self.yolo_ready),
            daemon = True
        )

        self.depth_process = mp.Process(
            target = depth_process,
            args = (self.depth_info_queue,
                    self.depth_results_queue,
                    self.run_event,
                    self.depth_ready),
            daemon = True
        )

        # Khởi động YOLO process trước
        self.yolo_process.start()
        print("Waiting for YOLO process to be ready...")
        yolo_ready_result = self.yolo_ready.wait(timeout=120.0)  # 2 phút cho model load
        print(f"YOLO ready: {yolo_ready_result}")
        
        # Sau đó khởi động Camera và Depth
        self.camera_process.start()
        self.depth_process.start()

        print("Waiting for other processes to be ready...")
        timeout = 30.0
        camera_ready_result = self.camera_ready.wait(timeout = timeout)
        depth_ready_result = self.depth_ready.wait(timeout = timeout)
        
        print(f"Camera ready: {camera_ready_result}")
        print(f"Depth ready: {depth_ready_result}")
        
        # Check if all processes are ready
        self.assertTrue(yolo_ready_result, "YOLO process failed to start")
        self.assertTrue(camera_ready_result, "Camera process failed to start")
        self.assertTrue(depth_ready_result, "Depth process failed to start")
        
        print("Collecting results...")
        detection_results = []
        depth_results = []
        max_results = 5 if not USE_CAMERA else 10  # Ít hơn cho ảnh tĩnh
        deadline = time.time() + (30 if not USE_CAMERA else 20)  # Lâu hơn cho ảnh tĩnh
        
        while time.time() < deadline and len(depth_results) < max_results:
            # Collect detection results
            try:
                result = self.detection_queue.get(timeout=0.5)
                if result:
                    frame, detections = result
                    detection_results.append(detections)
                    print(f"Get detection #{len(detection_results)}: {len(detections['bounding_boxes'])} object")
            except:
                pass
            
            # Collect depth results
            try:
                result = self.depth_results_queue.get(timeout=0.5)
                if result:
                    frame, depth_data = result
                    depth_results.append(depth_data)
                    print(f"Get depth #{len(depth_results)}: {len(depth_data)}  object")
                    
                    # Print depth data
                    for i, depth in enumerate(depth_data):
                        print(f" - Object {i+1}: Mean depth = {depth['mean_depth']:.2f}m")
            except:
                pass
            
            time.sleep(0.05)
            
        # Check final results:
        print(f"\nNumber of detections: {len(detection_results)}")
        print(f"Number of depth estimations: {len(depth_results)}")
        
        # Assertions
        self.assertGreater(len(detection_results), 0, "No detection results")
        
        # Chỉ kiểm tra depth nếu có bounding boxes
        has_bounding_boxes = any(len(d['bounding_boxes']) > 0 for d in detection_results)
        if not has_bounding_boxes:
            print("Warning: No objects detected in any frame")
        else:
            self.assertGreater(len(depth_results), 0, "No depth results despite having detected objects")
            # Kiểm tra chi tiết các kết quả depth
            for depth_result in depth_results:
                self.assertIsInstance(depth_result, list, "Depth result is not a list")
                if depth_result:
                    for obj in depth_result:
                        self.assertIn('mean_depth', obj, "Depth result does not contain mean_depth")
                        self.assertIn('min_depth', obj, "Depth result does not contain min_depth")
                        self.assertIn('max_depth', obj, "Depth result does not contain max_depth")
                    
        print("Test completed successfully")
        
if __name__ == "__main__":
    unittest.main()
        


