"""Cung cấp giao diện để tương tác với camera."""
import cv2
import numpy as np
import time
from typing import Tuple, Union, Optional


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
        
        # FPS tracking variables
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.last_fps_update = time.time()
        self.fps_update_interval = 1.0  # Cập nhật FPS mỗi giây

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
            
            # Apply optimized settings if enabled - matching optimized_realtime.py
            if self.use_optimized_settings:
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Giảm buffer để giảm lag
                # Chỉ dùng MJPG như trong optimized_realtime.py
                self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPG for better quality
            
            # Kiểm tra thông số camera thực tế
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
            
            if self.use_optimized_settings:
                actual_buffer = int(self.capture.get(cv2.CAP_PROP_BUFFERSIZE))
                print(f"✅ Camera setup: {actual_width}x{actual_height} @ {actual_fps:.1f}FPS (Buffer: {actual_buffer})")
                print("✅ Optimized settings applied (MJPG only, no exposure/autofocus changes)")
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
        
        # Cập nhật FPS counter
        self._update_fps()
        
        return frame

    def _update_fps(self) -> None:
        """Cập nhật thông tin FPS."""
        current_time = time.time()
        self.fps_counter += 1
        
        # Cập nhật FPS mỗi giây
        if current_time - self.last_fps_update >= self.fps_update_interval:
            time_elapsed = current_time - self.last_fps_update
            self.current_fps = self.fps_counter / time_elapsed
            self.fps_counter = 0
            self.last_fps_update = current_time

    def get_fps(self) -> float:
        """
        Lấy FPS hiện tại.
        
        Returns:
            float: FPS hiện tại
        """
        return self.current_fps

    def draw_fps(self, frame: np.ndarray, position: Tuple[int, int] = (10, 30), 
                 color: Tuple[int, int, int] = (0, 255, 0), font_scale: float = 1.0,
                 show_target_fps: bool = True) -> np.ndarray:
        """
        Vẽ thông tin FPS lên frame.
        
        Args:
            frame: Frame để vẽ FPS lên
            position: Vị trí vẽ text (x, y)
            color: Màu text (B, G, R)
            font_scale: Kích thước font
            show_target_fps: Có hiển thị target FPS không
            
        Returns:
            np.ndarray: Frame đã được vẽ FPS
        """
        result_frame = frame.copy()
        
        # Text hiển thị FPS hiện tại
        fps_text = f"FPS: {self.current_fps:.1f}"
        
        # Thêm target FPS nếu được yêu cầu
        if show_target_fps:
            fps_text += f" (Target: {self.fps})"
        
        # Vẽ background đen cho text để dễ đọc
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        cv2.rectangle(result_frame, 
                     (position[0] - 5, position[1] - text_size[1] - 5),
                     (position[0] + text_size[0] + 5, position[1] + 5),
                     (0, 0, 0), -1)
        
        # Vẽ text FPS
        cv2.putText(result_frame, fps_text, position, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        
        return result_frame

    def get_frame_with_fps(self, position: Tuple[int, int] = (10, 30), 
                          color: Tuple[int, int, int] = (0, 255, 0), 
                          font_scale: float = 1.0, show_target_fps: bool = True) -> np.ndarray:
        """
        Lấy frame đã được vẽ FPS.
        
        Args:
            position: Vị trí vẽ text (x, y)
            color: Màu text (B, G, R)
            font_scale: Kích thước font
            show_target_fps: Có hiển thị target FPS không
            
        Returns:
            np.ndarray: Frame đã được vẽ FPS
        """
        frame = self.get_frame()
        return self.draw_fps(frame, position, color, font_scale, show_target_fps)

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
            # Apply optimized settings - matching optimized_realtime.py
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            # Chỉ dùng MJPG, không thay đổi exposure hay autofocus
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            print("✅ Optimized settings enabled (MJPG only)")
        else:
            # Restore default settings
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 4)  # Default buffer
            # Reset về FOURCC mặc định
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
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
    
    print("Camera đang chạy với FPS tracking!")
    print("Các phím tắt:")
    print("  'q': Thoát")
    print("  'f': Bật/tắt hiển thị FPS")
    print("  's': In thông tin FPS ra console")
    
    show_fps = True
    try:
        while True:
            if show_fps:
                # Lấy frame với FPS được vẽ lên
                frame = camera.get_frame_with_fps(
                    position=(10, 30),
                    color=(0, 255, 0),  # Xanh lá
                    font_scale=0.8,
                    show_target_fps=True
                )
            else:
                # Lấy frame bình thường
                frame = camera.get_frame()
            
            cv2.imshow("Camera Feed with FPS", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                show_fps = not show_fps
                print(f"Hiển thị FPS: {'BẬT' if show_fps else 'TẮT'}")
            elif key == ord('s'):
                current_fps = camera.get_fps()
                print(f"FPS hiện tại: {current_fps:.2f} (Target: {camera.fps})")
                
    except KeyboardInterrupt:
        print("Đã dừng camera do nhận tín hiệu ngắt từ bàn phím")
    except Exception as e:
        print(f"Lỗi: {str(e)}")
    finally:
        camera.release() 