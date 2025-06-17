"""
Pipeline xử lý đa luồng cho camera, phát hiện đối tượng và ước tính độ sâu.
"""
import multiprocessing as mp
import time
import sys
import traceback
import threading
import queue
from typing import Any, Callable, Dict, List, Tuple, Optional, Union

# ---------------------- CLASS QUẢN LÝ QUEUE ĐƠN GIẢN ----------------------
class QueueManager:
    """Quản lý queue với cơ chế lấy giá trị mới nhất."""
    
    def __init__(self, maxsize: int = 10):
        """
        Khởi tạo Queue Manager.
        
        Args:
            maxsize: Kích thước tối đa của queue
        """
        self._queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
    
    def put(self, item: Any) -> None:
        """
        Thêm item vào queue, nếu đầy thì bỏ các item cũ.
        
        Args:
            item: Đối tượng cần thêm
        """
        with self._lock:
            try:
                # Nếu queue đầy, xóa item cũ nhất
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                # Thêm item mới
                self._queue.put_nowait(item)
            except queue.Full:
                pass  # Bỏ qua nếu không thể thêm
    
    def get(self, timeout: float = 0.1) -> Optional[Any]:
        """
        Lấy item mới nhất từ queue.
        
        Args:
            timeout: Thời gian tối đa chờ đợi (giây)
            
        Returns:
            Any: Item mới nhất hoặc None nếu không có
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_latest(self, timeout: float = 0.1) -> Optional[Any]:
        """
        Lấy item mới nhất từ queue (bỏ qua tất cả các item cũ).
        
        Args:
            timeout: Thời gian tối đa chờ đợi (giây)
            
        Returns:
            Any: Item mới nhất hoặc None nếu không có
        """
        with self._lock:
            # Thử lấy tất cả các item hiện có
            latest_item = None
            try:
                while True:
                    item = self._queue.get_nowait()
                    latest_item = item
            except queue.Empty:
                pass
            return latest_item


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
    """
    Worker chạy trong process con - chụp frame từ camera.
    
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
    """
    Worker chạy trong process con - phát hiện đối tượng bằng YOLO.
    
    Args:
        yolo_factory: Hàm factory tạo YOLO model
        frame_queue: Queue để lấy frame đầu vào
        detection_queue: Queue để đưa kết quả detection ra
        depth_info_queue: Queue để gửi thông tin tới depth process
        run_event: Event để báo hiệu process nên tiếp tục chạy
        detection_counter: Bộ đếm số lượng detection đã xử lý
        ready_event: Event báo hiệu process đã sẵn sàng
        error_queue: Queue để báo lỗi
        sleep: Thời gian sleep nếu không có frame mới
    """
    try:
        print(f"[YOLO Process {mp.current_process().pid}] Đã khởi động")
        
        try:
            # Import module division inside worker to avoid multiprocessing issues
            from detection import ModuleDivision
            
            yolo_model = yolo_factory()
            print(f"[YOLO Process {mp.current_process().pid}] Model đã khởi tạo thành công: {type(yolo_model).__name__}")
            divider = ModuleDivision()
            ready_event.set()
            
            detection_count = 0
            
            while run_event.is_set():
                try:
                    frame = frame_queue.get(timeout=1.0)
                    if frame is not None:
                        # Phát hiện đối tượng với YOLO
                        detections = yolo_model.detect(frame)
                        divided_result = divider.process_pallet_detections(detections, layer = 1)

                        depth_regions = divider.prepare_for_depth_estimation(divided_result)
                        
                        # Gửi kết quả detection ra ngoài
                        detection_queue.put((frame, detections))
                        
                        # Gửi thông tin cần thiết cho depth process (non-blocking)
                        depth_info = {
                            'frame': frame,
                            'regions': depth_regions,
                            'divided_result': divided_result,
                        }
                        
                        # Không chặn YOLO process nếu depth_info_queue đầy
                        try:
                            # Kiểm tra xem queue có đầy không
                            if depth_info_queue.full():
                                # Nếu đầy, bỏ qua việc đưa vào depth queue
                                # Điều này cho phép YOLO tiếp tục hoạt động
                                print(f"[YOLO Process] Depth queue đầy, bỏ qua xử lý depth cho frame này")
                            else:
                                # Sử dụng put_nowait thay vì put để tránh chặn
                                depth_info_queue.put_nowait(depth_info)
                        except Exception as e:
                            # Bỏ qua lỗi khi queue đầy
                            print(f"[YOLO Process] Không thể đưa vào depth queue: {str(e)}")
                        
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
    """
    Worker chạy trong process con - ước tính độ sâu.
    
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
        import threading
        import queue
        
        # Queue nội bộ trong process để tách việc nhận dữ liệu và xử lý depth
        internal_queue = queue.Queue(maxsize=5)
        
        # Flag để kiểm soát thread
        thread_running = threading.Event()
        thread_running.set()
        
        # Thread riêng để xử lý depth (tác vụ nặng)
        def depth_processing_thread():
            nonlocal depth_counter, depth_count
            print(f"[Depth Thread] Đã khởi động thread xử lý độ sâu")
            
            while thread_running.is_set():
                try:
                    # Lấy từ queue nội bộ (non-blocking)
                    try:
                        depth_task = internal_queue.get(timeout=0.5)
                        frame = depth_task['frame']
                        regions = depth_task.get('regions', [])
                        divided_result = depth_task.get('divided_result', {})
                        
                        # Xử lý depth cho từng region
                        depth_results = []
                        for region in regions:
                            bbox = region['bbox']
                            region_info = region['region_info']
                            
                            # Ước tính độ sâu cho bbox này
                            region_depth = depth_model.estimate_depth(frame, [bbox])
                            
                            # Tạo kết quả chi tiết cho region
                            if region_depth and len(region_depth) > 0:
                                depth_info = region_depth[0]  # Lấy kết quả đầu tiên
                                result = {
                                    'region_info': region_info,
                                    'bbox': bbox,
                                    'center': region['center'],
                                    'depth': depth_info,
                                    'position': {
                                        'x': region['center'][0],
                                        'y': region['center'][1], 
                                        'z': depth_info.get('mean_depth', 0.0) if isinstance(depth_info, dict) else 0.0
                                    }
                                }
                                depth_results.append(result)
                        
                        # Gửi kết quả ra
                        depth_result_queue.put((frame, depth_results))
                        
                        with depth_counter.get_lock():
                            depth_counter.value += 1
                        depth_count += 1
                        
                        if depth_count % 10 == 0:
                            print(f"[Depth Thread] Đã xử lý {depth_count} depth estimates cho {len(depth_results)} regions")
                    except queue.Empty:
                        time.sleep(0.01)
                        continue
                        
                except Exception as e:
                    error_msg = f"[Depth Thread] Lỗi khi xử lý depth: {str(e)}"
                    print(error_msg)
                    try:
                        error_queue.put(error_msg)
                    except:
                        pass
                    traceback.print_exc()
                    time.sleep(sleep)
        
        try:
            # Khởi tạo model
            depth_model = depth_factory()
            print(f"[Depth Process {mp.current_process().pid}] Model đã khởi tạo thành công: {type(depth_model).__name__}")
            
            # Báo hiệu đã sẵn sàng
            ready_event.set()
            
            depth_count = 0
            
            # Khởi động thread xử lý độ sâu
            depth_thread = threading.Thread(target=depth_processing_thread, daemon=True)
            depth_thread.start()
            
            # Vòng lặp chính trong process - chỉ nhận dữ liệu từ queue và chuyển vào queue nội bộ
            while run_event.is_set():
                try:
                    # Lấy dữ liệu từ queue liên process (có thể block nhưng chỉ trong thời gian ngắn)
                    try:
                        depth_info = depth_info_queue.get(timeout=0.1)
                        if depth_info is not None:
                            # Chỉ đưa vào queue nội bộ nếu nó không đầy (tránh tích tụ quá nhiều frame cũ)
                            if not internal_queue.full():
                                internal_queue.put(depth_info)
                            else:
                                # Nếu queue đầy, loại bỏ item cũ nhất và thêm item mới
                                try:
                                    internal_queue.get_nowait()  # Loại bỏ item cũ nhất
                                except queue.Empty:
                                    pass
                                internal_queue.put_nowait(depth_info)
                    except mp.queues.Empty:
                        time.sleep(sleep)
                        continue
                        
                except Exception as e:
                    error_msg = f"[Depth Process {mp.current_process().pid}] Lỗi khi nhận dữ liệu: {str(e)}"
                    print(error_msg)
                    error_queue.put(error_msg)
                    traceback.print_exc()
                    time.sleep(sleep)
                    
        except Exception as e:
            error_msg = f"[Depth Process {mp.current_process().pid}] Lỗi khởi tạo model: {str(e)}"
            print(error_msg)
            error_queue.put(error_msg)
            traceback.print_exc()
            
        finally:
            # Dừng thread xử lý
            thread_running.clear()
            if 'depth_thread' in locals() and depth_thread.is_alive():
                depth_thread.join(timeout=1.0)
                
    except Exception as e:
        error_msg = f"[Depth Process {mp.current_process().pid}] Lỗi khởi tạo process: {str(e)}"
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
        """
        Khởi tạo pipeline xử lý đa luồng.
        
        Args:
            camera_factory: Hàm factory tạo camera
            yolo_factory: Hàm factory tạo YOLO model
            depth_factory: Hàm factory tạo Depth model
            max_queue_size: Kích thước tối đa của mỗi queue
        """
        # Đảm bảo sử dụng spawn method cho Windows
        if mp.get_start_method(allow_none=True) != 'spawn':
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
        """
        Khởi động tất cả các process.
        
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
        """
        Lấy kết quả detection mới nhất.
        
        Args:
            timeout: Thời gian tối đa chờ đợi (giây)
            
        Returns:
            tuple: (frame, detections) hoặc None nếu không có kết quả mới
        """
        self._check_errors()
        return self.detection_manager.get(timeout=timeout)
    
    def get_depth(self, timeout: float = 0.1):
        """
        Lấy kết quả depth mới nhất.
        
        Args:
            timeout: Thời gian tối đa chờ đợi (giây)
            
        Returns:
            tuple: (frame, depth_results) hoặc None nếu không có kết quả mới
        """
        self._check_errors()
        return self.depth_manager.get(timeout=timeout)
    
    def get_latest_detection(self, timeout: float = 0.1):
        """
        Lấy kết quả detection mới nhất, bỏ qua các kết quả cũ.
        
        Args:
            timeout: Thời gian tối đa chờ đợi (giây)
            
        Returns:
            tuple: (frame, detections) hoặc None nếu không có kết quả mới
        """
        self._check_errors()
        return self.detection_manager.get_latest(timeout=timeout)
    
    def get_latest_depth(self, timeout: float = 0.1):
        """
        Lấy kết quả depth mới nhất, bỏ qua các kết quả cũ.
        
        Args:
            timeout: Thời gian tối đa chờ đợi (giây)
            
        Returns:
            tuple: (frame, depth_results) hoặc None nếu không có kết quả mới
        """
        self._check_errors()
        return self.depth_manager.get_latest(timeout=timeout)
    
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
    
    @property
    def is_running(self):
        """Kiểm tra xem pipeline có đang chạy không."""
        return (self._run_event.is_set() and 
                self._camera_process is not None and self._camera_process.is_alive() and
                self._yolo_process is not None and self._yolo_process.is_alive() and
                self._depth_process is not None and self._depth_process.is_alive())
    
    def get_stats(self):
        """
        Lấy thống kê về số lượng khung hình, detections và depth xử lý được.
        
        Returns:
            Dict: Thống kê về số lượng
                - 'frames': Số khung hình đã xử lý
                - 'detections': Số detections đã xử lý
                - 'depths': Số depth đã xử lý
        """
        return {
            'frames': self.frame_counter.value,
            'detections': self.detection_counter.value,
            'depths': self.depth_counter.value
        }


# Ví dụ sử dụng
if __name__ == "__main__":
    # Đây là ví dụ sử dụng, cần import các module thực tế
    from detection.camera import CameraInterface
    from detection.utils.yolo import YOLOInference
    from detection.utils.depth import DepthEstimator
    import cv2
    import time
    
    # Các factory functions
    def create_camera():
        camera = CameraInterface(camera_index=0)
        camera.initialize()
        return camera
    
    def create_yolo():
        return YOLOInference(model_path="best.pt", conf=0.25)
    
    def create_depth():
        return DepthEstimator()
    
    # Tạo pipeline
    pipeline = ProcessingPipeline(
        camera_factory=create_camera,
        yolo_factory=create_yolo,
        depth_factory=create_depth
    )
    
    # Khởi động pipeline
    if pipeline.start(timeout=60.0):
        print("Pipeline đã khởi động thành công!")
        
        try:
            # Lặp và xử lý kết quả
            for _ in range(100):  # Xử lý 100 khung hình
                # Lấy kết quả detection mới nhất
                detection_result = pipeline.get_latest_detection()
                if detection_result:
                    frame, detections = detection_result
                    # Xử lý kết quả detection
                    print(f"Đã phát hiện {len(detections.get('bounding_boxes', []))} đối tượng")
                
                # Lấy kết quả depth mới nhất
                depth_result = pipeline.get_latest_depth()
                if depth_result:
                    frame, depth_results = depth_result
                    # Xử lý kết quả depth
                    print(f"Đã ước tính độ sâu cho {len(depth_results)} đối tượng")
                
                # Hiển thị thống kê
                stats = pipeline.get_stats()
                print(f"Stats: Frames={stats['frames']}, Detections={stats['detections']}, Depths={stats['depths']}")
                
                time.sleep(0.1)  # Đợi một chút
                
        except KeyboardInterrupt:
            print("Đã nhận tín hiệu ngắt từ bàn phím")
        finally:
            # Dừng pipeline
            pipeline.stop()
            print("Pipeline đã dừng")
    else:
        print("Không thể khởi động pipeline!")
        # Kiểm tra lỗi
        for error in pipeline.errors:
            print(f"Lỗi: {error}") 