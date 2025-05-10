"""
Pipeline xử lý đa luồng với Camera, YOLO Detection và Depth Estimation
"""

import multiprocessing as mp
import time
import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple

from utility.queue_manager import QueueManager


# ---------------------- WORKER PROCESSES ----------------------

def _capture_frames_worker(
    camera_factory: Callable[[], Any],
    frame_queue: mp.Queue,
    run_event: mp.Event,
    frame_counter: mp.Value,
    ready_event: mp.Event,
    error_queue: mp.Queue,
    sleep: float = 0.01,
) -> None:
    """Worker chạy trong process con - chụp frame từ camera.
    
    Args:
        camera_factory: Hàm factory tạo camera
        frame_queue: Queue để đưa frame vào
        run_event: Event để báo hiệu process nên tiếp tục chạy
        frame_counter: Bộ đếm frame đã xử lý
        ready_event: Event báo hiệu process đã sẵn sàng
        error_queue: Queue để báo lỗi
        sleep: Thời gian sleep giữa các lần lấy frame nếu không có frame mới
    """
    try:
        print(f"[Camera Process {mp.current_process().pid}] Đã khởi động")
        
        try:
            camera = camera_factory()
            print(f"[Camera Process {mp.current_process().pid}] Camera đã khởi tạo thành công: {type(camera).__name__}")
            
            # Đảm bảo ready_event được set
            if not ready_event.is_set():
                ready_event.set()
                print(f"[Camera Process {mp.current_process().pid}] Đã đặt ready_event")
            
            frame_count = 0
            
            while run_event.is_set():
                try:
                    frame = camera.get_frame()
                    if frame is not None:
                        frame_queue.put(frame)
                        with frame_counter.get_lock():
                            frame_counter.value += 1
                        frame_count += 1
                        if frame_count % 10 == 0:
                            print(f"[Camera Process {mp.current_process().pid}] Đã xử lý {frame_count} frames")
                    else:
                        time.sleep(sleep)
                except Exception as e:
                    error_msg = f"[Camera Process {mp.current_process().pid}] Lỗi khi xử lý frame: {e}"
                    print(error_msg)
                    error_queue.put(error_msg)
                    traceback.print_exc()
                    time.sleep(sleep)
        except Exception as e:
            error_msg = f"[Camera Process {mp.current_process().pid}] Lỗi khởi tạo camera: {e}"
            print(error_msg)
            error_queue.put(error_msg)
            traceback.print_exc()
    except Exception as e:
        error_msg = f"[Camera Process {mp.current_process().pid}] Lỗi khởi tạo process: {e}"
        print(error_msg)
        try:
            error_queue.put(error_msg)
        except:
            pass
        traceback.print_exc()
    finally:
        # Đảm bảo ready_event được set dù có lỗi
        if not ready_event.is_set():
            ready_event.set()
            print(f"[Camera Process {mp.current_process().pid}] Đã đặt ready_event (finally)")


def _yolo_detection_worker(
    yolo_factory: Callable[[], Any],
    frame_queue: mp.Queue,
    detection_queue: mp.Queue,
    depth_info_queue: mp.Queue,  # Queue để gửi thông tin tới depth process
    run_event: mp.Event,
    detection_counter: mp.Value,
    ready_event: mp.Event,
    error_queue: mp.Queue,
    sleep: float = 0.01,
) -> None:
    """Worker chạy trong process con - phát hiện đối tượng bằng YOLO.
    
    Args:
        yolo_factory: Hàm factory tạo YOLO model
        frame_queue: Queue để lấy frame từ camera
        detection_queue: Queue để đưa kết quả phát hiện ra
        depth_info_queue: Queue để gửi thông tin cho depth process
        run_event: Event để báo hiệu process nên tiếp tục chạy 
        detection_counter: Bộ đếm số lượng detection đã xử lý
        ready_event: Event báo hiệu process đã sẵn sàng
        error_queue: Queue để báo lỗi
        sleep: Thời gian sleep giữa các lần lấy frame nếu không có frame mới
    """
    try:
        print(f"[YOLO Process {mp.current_process().pid}] Đã khởi động")
        
        try:
            yolo_model = yolo_factory()
            print(f"[YOLO Process {mp.current_process().pid}] Model đã khởi tạo thành công: {type(yolo_model).__name__}")
            
            ready_event.set()
            
            detection_count = 0
            
            while run_event.is_set():
                try:
                    frame = frame_queue.get(timeout=0.5)
                    if frame is not None:
                        # Phát hiện đối tượng với YOLO
                        detections = yolo_model.detect(frame)
                        
                        # Gửi kết quả detection ra ngoài
                        detection_queue.put((frame, detections))
                        
                        # Gửi thông tin cần thiết cho depth process
                        depth_info = {
                            'frame': frame,
                            'bounding_boxes': detections.get('bounding_boxes', [])
                        }
                        depth_info_queue.put(depth_info)
                        
                        with detection_counter.get_lock():
                            detection_counter.value += 1
                        detection_count += 1
                        
                        if detection_count % 10 == 0:
                            print(f"[YOLO Process {mp.current_process().pid}] Đã xử lý {detection_count} detections")
                    else:
                        time.sleep(sleep)
                except Exception as e:
                    error_msg = f"[YOLO Process {mp.current_process().pid}] Lỗi khi xử lý detection: {e}"
                    print(error_msg)
                    error_queue.put(error_msg)
                    traceback.print_exc()
                    time.sleep(sleep)
        except Exception as e:
            error_msg = f"[YOLO Process {mp.current_process().pid}] Lỗi khởi tạo model: {e}"
            print(error_msg)
            error_queue.put(error_msg)
            traceback.print_exc()
    except Exception as e:
        error_msg = f"[YOLO Process {mp.current_process().pid}] Lỗi khởi tạo process: {e}"
        print(error_msg)
        try:
            error_queue.put(error_msg)
        except:
            pass
        traceback.print_exc()
    finally:
        ready_event.set()


def _depth_estimation_worker(
    depth_factory: Callable[[], Any],
    depth_info_queue: mp.Queue,
    depth_result_queue: mp.Queue,
    run_event: mp.Event,
    depth_counter: mp.Value,
    ready_event: mp.Event,
    error_queue: mp.Queue,
    sleep: float = 0.01,
) -> None:
    """Worker chạy trong process con - ước tính độ sâu.
    
    Args:
        depth_factory: Hàm factory tạo Depth model
        depth_info_queue: Queue để lấy thông tin từ YOLO process
        depth_result_queue: Queue để đưa kết quả độ sâu ra
        run_event: Event để báo hiệu process nên tiếp tục chạy
        depth_counter: Bộ đếm số lượng depth đã xử lý
        ready_event: Event báo hiệu process đã sẵn sàng
        error_queue: Queue để báo lỗi
        sleep: Thời gian sleep nếu không có dữ liệu mới
    """
    try:
        print(f"[Depth Process {mp.current_process().pid}] Đã khởi động")
        
        try:
            depth_model = depth_factory()
            print(f"[Depth Process {mp.current_process().pid}] Model đã khởi tạo thành công: {type(depth_model).__name__}")
            
            ready_event.set()
            
            depth_count = 0
            
            while run_event.is_set():
                try:
                    depth_info = depth_info_queue.get(timeout=0.5)
                    if depth_info is not None:
                        frame = depth_info['frame']
                        bounding_boxes = depth_info['bounding_boxes']
                        
                        # Ước tính độ sâu cho các bounding box
                        depth_results = depth_model.estimate_depth(frame, bounding_boxes)
                        
                        # Gửi kết quả ra
                        depth_result_queue.put((frame, depth_results))
                        
                        with depth_counter.get_lock():
                            depth_counter.value += 1
                        depth_count += 1
                        
                        if depth_count % 10 == 0:
                            print(f"[Depth Process {mp.current_process().pid}] Đã xử lý {depth_count} depth estimates")
                    else:
                        time.sleep(sleep)
                except Exception as e:
                    error_msg = f"[Depth Process {mp.current_process().pid}] Lỗi khi xử lý depth: {e}"
                    print(error_msg)
                    error_queue.put(error_msg)
                    traceback.print_exc()
                    time.sleep(sleep)
        except Exception as e:
            error_msg = f"[Depth Process {mp.current_process().pid}] Lỗi khởi tạo model: {e}"
            print(error_msg)
            error_queue.put(error_msg)
            traceback.print_exc()
    except Exception as e:
        error_msg = f"[Depth Process {mp.current_process().pid}] Lỗi khởi tạo process: {e}"
        print(error_msg)
        try:
            error_queue.put(error_msg)
        except:
            pass
        traceback.print_exc()
    finally:
        ready_event.set()


# ---------------------- MAIN CLASS ----------------------

class ProcessingPipeline:
    """Pipeline xử lý đa luồng với Camera, YOLO và Depth Estimation."""
    
    def __init__(
        self,
        camera_factory: Callable[[], Any],
        yolo_factory: Callable[[], Any],
        depth_factory: Callable[[], Any],
        max_queue_size: int = 10
    ):
        """Khởi tạo pipeline xử lý đa luồng.
        
        Args:
            camera_factory: Hàm factory tạo camera
            yolo_factory: Hàm factory tạo YOLO model
            depth_factory: Hàm factory tạo Depth model
            max_queue_size: Kích thước tối đa của mỗi queue
        """
        # Đảm bảo sử dụng spawn method cho Windows
        if mp.get_start_method() != 'spawn':
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
        
        # Lưu các factory function
        self._camera_factory = camera_factory
        self._yolo_factory = yolo_factory
        self._depth_factory = depth_factory
        
        # Tạo các queue
        self._frame_queue = mp.Queue(maxsize=max_queue_size)
        self._detection_queue = mp.Queue(maxsize=max_queue_size)
        self._depth_info_queue = mp.Queue(maxsize=max_queue_size)
        self._depth_result_queue = mp.Queue(maxsize=max_queue_size)
        self._error_queue = mp.Queue()
        
        # Tạo các bộ đếm
        self.frame_counter = mp.Value('i', 0)
        self.detection_counter = mp.Value('i', 0)
        self.depth_counter = mp.Value('i', 0)
        
        # Tạo các event
        self._run_event = mp.Event()
        self._camera_ready_event = mp.Event()
        self._yolo_ready_event = mp.Event()
        self._depth_ready_event = mp.Event()
        
        # Các process
        self._camera_process = None
        self._yolo_process = None
        self._depth_process = None
        
        # Danh sách lỗi
        self._errors = []
        
        # Các QueueManager cho việc lấy kết quả
        self.detection_manager = QueueManager(maxsize=max_queue_size)
        self.depth_manager = QueueManager(maxsize=max_queue_size)
    
    def start(self, timeout: float = 30.0) -> bool:
        """Khởi động tất cả các process.
        
        Args:
            timeout: Thời gian tối đa chờ các process khởi động (giây)
            
        Returns:
            bool: True nếu tất cả process đã sẵn sàng, False nếu không
        """
        # Đặt run event
        self._run_event.set()
        
        # Clear các ready event
        self._camera_ready_event.clear()
        self._yolo_ready_event.clear() 
        self._depth_ready_event.clear()
        
        # Khởi động Camera Process
        self._camera_process = mp.Process(
            target=_capture_frames_worker,
            args=(
                self._camera_factory,
                self._frame_queue,
                self._run_event,
                self.frame_counter,
                self._camera_ready_event,
                self._error_queue,
            ),
            daemon=True,
        )
        self._camera_process.start()
        print(f"Camera Process đã khởi động với PID: {self._camera_process.pid}")
        
        # Khởi động YOLO Process
        self._yolo_process = mp.Process(
            target=_yolo_detection_worker,
            args=(
                self._yolo_factory,
                self._frame_queue,
                self._detection_queue,
                self._depth_info_queue,
                self._run_event,
                self.detection_counter,
                self._yolo_ready_event,
                self._error_queue,
            ),
            daemon=True,
        )
        self._yolo_process.start()
        print(f"YOLO Process đã khởi động với PID: {self._yolo_process.pid}")
        
        # Khởi động Depth Process
        self._depth_process = mp.Process(
            target=_depth_estimation_worker,
            args=(
                self._depth_factory,
                self._depth_info_queue,
                self._depth_result_queue,
                self._run_event,
                self.depth_counter,
                self._depth_ready_event,
                self._error_queue,
            ),
            daemon=True,
        )
        self._depth_process.start()
        print(f"Depth Process đã khởi động với PID: {self._depth_process.pid}")
        
        # Đợi camera process sẵn sàng với timeout dài hơn vì thường khởi tạo camera mất nhiều thời gian
        print(f"Đang đợi Camera Process (tối đa {timeout}s)...")
        camera_ready = self._camera_ready_event.wait(timeout)
        
        # Đợi các process khác
        yolo_ready = self._yolo_ready_event.wait(timeout)
        depth_ready = self._depth_ready_event.wait(timeout)
        
        # Kiểm tra lỗi
        self._check_errors()
        
        # In thông tin trạng thái
        print(f"Camera Process ready: {camera_ready}")
        print(f"YOLO Process ready: {yolo_ready}")
        print(f"Depth Process ready: {depth_ready}")
        
        # Nếu phát hiện camera process còn sống nhưng không sẵn sàng, thử kiểm tra lại một lần nữa
        if not camera_ready and self._camera_process.is_alive():
            print("Camera process còn sống nhưng chưa sẵn sàng, kiểm tra lại...")
            # Thử kiểm tra lại nếu Process còn sống
            time.sleep(1.0)  # Đợi thêm chút nữa
            camera_ready = self._camera_ready_event.is_set()
            print(f"Kiểm tra lại Camera Process ready: {camera_ready}")
        
        # Khởi động background thread để chuyển kết quả từ Queue vào QueueManager
        if camera_ready and yolo_ready and depth_ready:
            self._start_queue_workers()
            return True
        else:
            # In thông tin debug về các process đang chạy
            print(f"Camera Process is alive: {self._camera_process.is_alive()}")
            print(f"YOLO Process is alive: {self._yolo_process.is_alive()}")
            print(f"Depth Process is alive: {self._depth_process.is_alive()}")
            return False
    
    def _start_queue_workers(self):
        """Khởi động các thread để chuyển dữ liệu từ Queue sang QueueManager."""
        import threading
        
        # Thread để chuyển kết quả detection
        def detection_worker():
            while self._run_event.is_set():
                try:
                    result = self._detection_queue.get(timeout=0.5)
                    if result is not None:
                        self.detection_manager.put(result)
                except:
                    pass
        
        # Thread để chuyển kết quả depth
        def depth_worker():
            while self._run_event.is_set():
                try:
                    result = self._depth_result_queue.get(timeout=0.5)
                    if result is not None:
                        self.depth_manager.put(result)
                except:
                    pass
        
        # Khởi động các thread
        threading.Thread(target=detection_worker, daemon=True).start()
        threading.Thread(target=depth_worker, daemon=True).start()
    
    def stop(self):
        """Dừng tất cả các process."""
        print("Đang dừng tất cả các process...")
        self._run_event.clear()
        
        # Dừng và join từng process
        if self._camera_process:
            self._camera_process.join(timeout=2)
            print(f"Camera Process đã dừng, exit code: {self._camera_process.exitcode}")
            self._camera_process = None
            
        if self._yolo_process:
            self._yolo_process.join(timeout=2)
            print(f"YOLO Process đã dừng, exit code: {self._yolo_process.exitcode}")
            self._yolo_process = None
            
        if self._depth_process:
            self._depth_process.join(timeout=2)
            print(f"Depth Process đã dừng, exit code: {self._depth_process.exitcode}")
            self._depth_process = None
    
    def get_detection(self, timeout: float = 0.1):
        """Lấy kết quả detection mới nhất.
        
        Returns:
            tuple: (frame, detections) hoặc None nếu không có kết quả mới
        """
        self._check_errors()
        return self.detection_manager.get(timeout=timeout)
    
    def get_depth(self, timeout: float = 0.1):
        """Lấy kết quả depth mới nhất.
        
        Returns:
            tuple: (frame, depth_results) hoặc None nếu không có kết quả mới
        """
        self._check_errors()
        return self.depth_manager.get(timeout=timeout)
    
    def _check_errors(self):
        """Kiểm tra và lưu lỗi từ các process con."""
        while not self._error_queue.empty():
            try:
                error = self._error_queue.get_nowait()
                self._errors.append(error)
                print(f"Lỗi từ process con: {error}")
            except:
                break
    
    @property
    def errors(self):
        """Trả về danh sách lỗi từ các process con."""
        self._check_errors()
        return self._errors 