"""Cung cấp giao diện để tương tác với camera."""
import cv2
import numpy as np
from typing import Tuple, Union


class CameraInterface:
    """Giao diện tương tác với camera vật lý hoặc camera mạng."""

    def __init__(self, camera_index: Union[int, str] = 0, resolution: Tuple[int, int] = (1280, 1024), fps: int = 30, 
                 use_optimized_settings: bool = True):
        """
        Khởi tạo giao diện camera.

        Args:
            camera_index: Index của camera hoặc địa chỉ URL cho camera mạng
            resolution: Độ phân giải (width, height) mong muốn
            fps: Số khung hình mỗi giây mong muốn
            use_optimized_settings: Có sử dụng optimized settings hay không
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.use_optimized_settings = use_optimized_settings
        self.capture = None
        self.is_initialized = False

    def initialize(self) -> None:
        """
        Khởi tạo và mở kết nối đến camera với optimized settings.
        
        Raises:
            RuntimeError: Nếu không thể mở camera
        """
        try:
            self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                # Thử index camera khác nếu mặc định không hoạt động
                self.capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Thử index 1
                if not self.capture.isOpened():
                    raise RuntimeError(f"Không thể mở camera tại index {self.camera_index} hoặc 1")
            
            # Basic camera settings
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Apply optimized settings if enabled
            if self.use_optimized_settings:
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Giảm buffer để giảm lag
                
                # Tối ưu thêm cho Logitech cameras và các camera tương tự
                self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure
                self.capture.set(cv2.CAP_PROP_EXPOSURE, -6)         # Fast exposure
                self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)         # Disable autofocus
                self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPG for better performance
            
            # Kiểm tra thông số camera thực tế
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
            
            if self.use_optimized_settings:
                actual_buffer = int(self.capture.get(cv2.CAP_PROP_BUFFERSIZE))
                print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}FPS (Buffer: {actual_buffer})")
                print("✅ Optimized camera settings applied")
            else:
                print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}FPS")
                print("📹 Standard camera settings")
            
            self.is_initialized = True
        except Exception as e:
            self.is_initialized = False
            raise RuntimeError(f"Lỗi khởi tạo camera: {str(e)}")

    def get_frame(self) -> np.ndarray:
        """
        Lấy khung hình từ camera.
        
        Returns:
            np.ndarray: Khung hình dạng numpy array với định dạng BGR
            
        Raises:
            RuntimeError: Nếu camera chưa được khởi tạo hoặc không thể lấy khung hình
        """
        if not self.is_initialized or self.capture is None:
            raise RuntimeError("Camera chưa được khởi tạo")
            
        ret, frame = self.capture.read()
        if not ret:
            raise RuntimeError("Không thể đọc khung hình từ camera")
        return frame

    def get_resolution(self) -> Tuple[int, int]:
        """
        Lấy độ phân giải thực tế của camera.
        
        Returns:
            Tuple[int, int]: (width, height) thực tế của camera
        """
        if not self.is_initialized or self.capture is None:
            return (0, 0)
            
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def set_optimized_settings(self, enabled: bool) -> None:
        """
        Bật/tắt optimized settings cho camera đã khởi tạo.
        
        Args:
            enabled: True để bật optimized settings, False để tắt
        """
        if not self.is_initialized or self.capture is None:
            print("⚠️ Camera chưa được khởi tạo")
            return
            
        self.use_optimized_settings = enabled
        
        if enabled:
            # Apply optimized settings
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.capture.set(cv2.CAP_PROP_EXPOSURE, -6)
            self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            print("✅ Optimized settings enabled")
        else:
            # Restore default settings
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 4)  # Default buffer
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Enable auto exposure
            self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
            print("📹 Standard settings restored")

    def release(self) -> None:
        """Giải phóng tài nguyên camera."""
        if self.capture is not None:
            self.capture.release()
            cv2.destroyAllWindows()
            self.is_initialized = False


if __name__ == "__main__":
    camera = CameraInterface()
    camera.initialize()
    try:
        while True:
            frame = camera.get_frame()
            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Đã dừng camera do nhận tín hiệu ngắt từ bàn phím")
    except Exception as e:
        print(f"Lỗi: {str(e)}")
    finally:
        camera.release() 