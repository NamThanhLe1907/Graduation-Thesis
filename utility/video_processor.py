"""
FrameCamera – spawn một process riêng để lấy frame liên tục
bằng *camera_factory* và đẩy vào QueueManager.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
import sys
from typing import Any, Callable

from utility.archived.queue_manager import QueueManager

# ---------------------------------------------------------------------
# QUAN TRỌNG: Worker function PHẢI ở top-level module để hoạt động với Windows
def _capture_frames(
    camera_factory: Callable[[], Any],
    raw_queue: mp.Queue,
    run_event: mp.Event,
    frame_counter: mp.Value,
    ready_event: mp.Event,  # Thêm sự kiện để biết khi process đã sẵn sàng
    error_queue: mp.Queue,  # Thêm queue để gửi lỗi về process cha
    sleep: float = 0.01,
) -> None:
    """Worker chạy trong process con.
    
    Lưu ý quan trọng: function này phải ở top-level module để có thể pickle
    trên Windows.
    """
    try:
        print(f"[Process {mp.current_process().pid}] Process con đã khởi động")
        
        try:
            print(f"[Process {mp.current_process().pid}] Đang khởi tạo camera...")
            camera = camera_factory()  # KHỞI TẠO camera TRONG process con
            print(f"[Process {mp.current_process().pid}] Camera đã khởi tạo thành công: {type(camera).__name__}")
            
            # Báo hiệu process đã sẵn sàng
            ready_event.set()
            print(f"[Process {mp.current_process().pid}] Đã đặt ready_event")
            
            frame_count = 0
            
            while run_event.is_set():
                try:
                    frame = camera.get_frame()
                    if frame is not None:
                        raw_queue.put(frame)
                        with frame_counter.get_lock():
                            frame_counter.value += 1
                        frame_count += 1
                        if frame_count % 10 == 0:
                            print(f"[Process {mp.current_process().pid}] Đã xử lý {frame_count} frames")
                    else:
                        time.sleep(sleep)
                except Exception as e:
                    error_msg = f"[Process {mp.current_process().pid}] Lỗi khi xử lý frame: {e}"
                    print(error_msg)
                    error_queue.put(error_msg)
                    traceback.print_exc()
                    time.sleep(sleep)
        except Exception as e:
            error_msg = f"[Process {mp.current_process().pid}] Lỗi khởi tạo camera: {e}"
            print(error_msg)
            error_queue.put(error_msg)
            traceback.print_exc()
    except Exception as e:
        error_msg = f"[Process {mp.current_process().pid}] Lỗi khởi tạo process con: {e}"
        print(error_msg)
        try:
            error_queue.put(error_msg)
        except:
            pass
        traceback.print_exc()
    finally:
        # Đảm bảo ready_event được set dù có lỗi
        ready_event.set()
        print(f"[Process {mp.current_process().pid}] Đã đặt ready_event (finally)")
# ---------------------------------------------------------------------


class FrameCamera:
    """Đọc camera ở process con, lưu frame mới nhất vào queue."""

    def __init__(self, camera_factory: Callable[[], Any], max_queue: int = 30):
        """
        Parameters
        ----------
        camera_factory : Callable[[], Any]
            Hàm không tham số, trả về object có phương thức ``get_frame()``.
            (Ví dụ: ``lambda: CameraInterface(0)`` hoặc ``lambda: MockCamera()``)
            
            LƯU Ý: Trên Windows, camera_factory phải là một function ở top-level 
            module (không phải lambda hoặc local function). Nên dùng như:
            
            ```python
            def create_camera():
                return MockCamera()
                
            cam = FrameCamera(create_camera)
            ```
            
        max_queue : int
            Số khung hình bộ đệm (không nên quá lớn để tránh trễ hình).
        """
        # Đảm bảo chúng ta đang dùng phương thức spawn cho Windows
        if mp.get_start_method() != 'spawn':
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                # Có thể đã đặt ở nơi khác trong ứng dụng
                pass
            
        self._camera_factory = camera_factory
        self.frame_queue = QueueManager(maxsize=max_queue)

        self.frame_counter: mp.Value = mp.Value("i", 0)
        self._run_event = mp.Event()
        self._ready_event = mp.Event()  # Thêm sự kiện để biết khi process đã sẵn sàng
        self._error_queue = mp.Queue()  # Thêm queue để nhận lỗi từ process con
        self._proc: mp.Process | None = None
        self._errors = []

    # ---------------- control ----------------
    def start(self, timeout: float = 5.0) -> bool:
        """Khởi động process con. 
        
        Returns:
            bool: True nếu process khởi động thành công và đã sẵn sàng, False nếu không
        """
        if self._proc and self._proc.is_alive():
            return True
        
        self._run_event.set()
        self._ready_event.clear()
        
        self._proc = mp.Process(
            target=_capture_frames,
            args=(
                self._camera_factory,
                self.frame_queue._q,   # raw multiprocessing.Queue
                self._run_event,
                self.frame_counter,
                self._ready_event,      # Truyền sự kiện ready
                self._error_queue,      # Truyền queue lỗi
            ),
            daemon=True,
        )
        self._proc.start()
        print(f"Process đã khởi động với PID: {self._proc.pid}")
        
        # Đợi process con báo hiệu đã sẵn sàng
        ready = self._ready_event.wait(timeout)
        
        # Kiểm tra có lỗi nào từ process con không
        self._check_errors()
        
        if ready:
            print("Process con đã sẵn sàng")
        else:
            print("Hết thời gian chờ process con - có thể chưa sẵn sàng")
            
            # In ra thông tin debug nếu process vẫn còn sống nhưng không ready
            if self._proc.is_alive():
                print(f"Process còn sống (PID: {self._proc.pid}) nhưng không sẵn sàng")
            else:
                print(f"Process đã chết (PID: {self._proc.pid})")
        
        return ready

    def stop(self) -> None:
        print("Đang dừng process...")
        self._run_event.clear()
        if self._proc:
            self._proc.join(timeout=2)
            print(f"Process đã dừng, exit code: {self._proc.exitcode}")
            self._proc = None

    # ---------------- API ----------------
    def get_frame(self, timeout: float = 0.1):
        """Trả về frame mới nhất hoặc ``None`` sau *timeout* giây."""
        self._check_errors()  # Kiểm tra lỗi mới trước khi get frame
        return self.frame_queue.get(timeout=timeout)
    
    def _check_errors(self):
        """Kiểm tra và lưu lỗi từ process con"""
        while not self._error_queue.empty():
            try:
                error = self._error_queue.get_nowait()
                self._errors.append(error)
                print(f"Lỗi từ process con: {error}")
            except:
                break
    
    @property
    def errors(self):
        """Trả về danh sách lỗi từ process con"""
        self._check_errors()
        return self._errors

