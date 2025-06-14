"""
Mock FrameCamera sử dụng threading thay vì multiprocessing để dễ dàng testing
"""
import threading
import time
from queue import Queue
from typing import Any, Callable, List, Optional


class MockFrameCamera:
    """Phiên bản đơn giản của FrameCamera dùng threading cho testing"""
    
    def __init__(self, camera_factory: Callable[[], Any], max_queue: int = 30):
        """
        Parameters
        ----------
        camera_factory : Callable[[], Any]
            Hàm tạo camera mock
        max_queue : int
            Kích thước queue
        """
        self._camera_factory = camera_factory
        self._queue = Queue(maxsize=max_queue)
        self._run_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frame_counter = 0
        self.frames_captured: List[str] = []  # Lưu các frame đã capture để debug
    
    def _worker(self):
        """Thread worker để lấy frames từ camera"""
        print("Thread worker đã khởi động")
        camera = self._camera_factory()
        print(f"Camera đã khởi tạo thành công: {type(camera).__name__}")
        
        while self._run_event.is_set():
            try:
                frame = camera.get_frame()
                if frame is not None:
                    # Nếu queue đầy, loại bỏ frame cũ nhất
                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                        except:
                            pass
                    
                    self._queue.put(frame)
                    self.frame_counter += 1
                    self.frames_captured.append(frame)  # Lưu lại để debug
                    print(f"Đã bỏ frame {frame} vào queue")
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Lỗi khi xử lý frame: {e}")
                time.sleep(0.01)
    
    def start(self):
        """Khởi động worker thread"""
        if self._thread and self._thread.is_alive():
            return
        
        self._run_event.set()
        self._thread = threading.Thread(target=self._worker)
        self._thread.daemon = True
        self._thread.start()
        print(f"Worker thread đã khởi động")
        
        # Đợi một chút để thread có thể khởi động
        time.sleep(0.5)
        return True
    
    def stop(self):
        """Dừng worker thread"""
        print("Đang dừng thread...")
        self._run_event.clear()
        if self._thread:
            self._thread.join(timeout=2)
            print("Thread đã dừng")
    
    def get_frame(self, timeout=0.1):
        """Lấy frame mới nhất từ queue"""
        try:
            return self._queue.get(timeout=timeout)
        except:
            return None 