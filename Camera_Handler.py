from __future__ import annotations
import cv2
import threading
import time
from typing import Optional, Tuple
from dataclasses import dataclass
from src.logger import AppLogger, PerformanceMetrics

logger = AppLogger.get_logger("camera")

@dataclass(frozen=True)
class CameraConfig:
    """Cấu hình camera với giá trị mặc định"""
    source: int = 0
    resolution: Tuple[int, int] = (1280, 1024)
    api_preference: int = cv2.CAP_DSHOW
    buffer_size: int = 10
    timeout_ms: int = 2000

class CameraError(Exception):
    """Base exception cho các lỗi camera"""
    def __init__(self, message: str, error_code: Optional[int] = None):
        super().__init__(message)
        self.error_code = error_code

class CameraTimeoutError(CameraError):
    """Lỗi timeout khi đọc frame từ camera"""

class ICamera:
    """Interface cho các loại camera"""
    def capture_frame(self) -> Optional[cv2.Mat]:
        raise NotImplementedError
        
    def release(self) -> None:
        raise NotImplementedError

class HDWebCamera(ICamera):
    """Triển khai camera HD với thread-safe buffer"""
    
    def __init__(self, config: CameraConfig):
        self._validate_config(config)
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_buffer: Optional[cv2.Mat] = None
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        
        self._init_camera()
        self._start_capture_thread()

    def _validate_config(self, config: CameraConfig) -> None:
        """Validate cấu hình camera"""
        if not isinstance(config.resolution, tuple) or len(config.resolution) != 2:
            raise ValueError("Độ phân giải phải là tuple (width, height)")
        if config.buffer_size < 1:
            raise ValueError("Kích thước buffer phải >= 1")

    def _init_camera(self) -> None:
        """Khởi tạo camera với cấu hình đã cho"""
        try:
            self._cap = cv2.VideoCapture(
                self.config.source,
                self.config.api_preference
            )
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            
            if not self._cap.isOpened():
                raise CameraError("Không thể kết nối camera")

        except Exception as e:
            logger.error(f"Lỗi khởi tạo camera: {str(e)}")
            raise CameraError(f"Lỗi khởi tạo camera: {str(e)}") from e

    def _start_capture_thread(self) -> None:
        """Khởi động thread đọc frame liên tục"""
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_frames,
            daemon=True
        )
        self._capture_thread.start()
        logger.info("Camera capture thread started")

    def _capture_frames(self) -> None:
        """Luồng chính đọc frame từ camera"""
        while self._running and self._cap.isOpened():
            try:
                ret, frame = self._cap.read()
                if ret:
                    self._frame_buffer = frame.copy()
                else:
                    logger.warning("Không đọc được frame từ camera")
            except Exception as e:
                logger.error(f"Lỗi đọc frame: {str(e)}")
                self._running = False

    def capture_frame(self) -> Optional[cv2.Mat]:
        """Lấy frame mới nhất từ buffer"""
        if self._frame_buffer is None:
            logger.debug("Buffer trống")
            return None
            
        try:
            return self._frame_buffer.copy()
        except Exception as e:
            logger.error(f"Lỗi sao chép frame: {str(e)}")
            return None

    def release(self) -> None:
        """Giải phóng tài nguyên camera"""
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2)
            logger.info("Camera thread stopped")
            
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Camera resources released")
