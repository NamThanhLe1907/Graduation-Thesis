"""
Cung cấp công cụ theo dõi hiệu suất hệ thống.
"""
import time
import statistics
from typing import Dict, List, Optional
import threading
import datetime


class PerformanceMonitor:
    """Theo dõi hiệu suất xử lý khung hình và suy luận."""
    
    def __init__(self, max_history: int = 100):
        """
        Khởi tạo bộ theo dõi hiệu suất.
        
        Args:
            max_history: Số lượng tối đa các mẫu lưu trong lịch sử
        """
        self.start_time = time.time()
        self.frames_processed = 0
        self.total_inference_time = 0
        self.frame_start_time = 0
        self.max_history = max_history
        
        # Lưu lịch sử để tính toán thống kê
        self.inference_times: List[float] = []
        
        # Theo dõi chi tiết theo từng stage
        self.stage_times: Dict[str, List[float]] = {}
        self.current_stage: Optional[str] = None
        self.stage_start_time = 0
        
        # Lock để đảm bảo thread-safety
        self._lock = threading.Lock()
    
    def start_frame(self):
        """Bắt đầu đo thời gian cho một khung hình mới."""
        with self._lock:
            self.frame_start_time = time.time()
    
    def end_frame(self):
        """Kết thúc đo thời gian và cập nhật chỉ số."""
        with self._lock:
            elapsed = time.time() - self.frame_start_time
            self.frames_processed += 1
            self.total_inference_time += elapsed
            
            # Thêm vào lịch sử, giới hạn kích thước
            self.inference_times.append(elapsed)
            if len(self.inference_times) > self.max_history:
                self.inference_times.pop(0)
    
    def start_stage(self, stage_name: str):
        """
        Bắt đầu đo thời gian cho một giai đoạn cụ thể.
        
        Args:
            stage_name: Tên của giai đoạn (ví dụ: 'detection', 'depth')
        """
        with self._lock:
            self.current_stage = stage_name
            self.stage_start_time = time.time()
    
    def end_stage(self):
        """Kết thúc đo thời gian cho giai đoạn hiện tại."""
        with self._lock:
            if self.current_stage is None:
                return
                
            elapsed = time.time() - self.stage_start_time
            
            # Khởi tạo danh sách nếu chưa tồn tại
            if self.current_stage not in self.stage_times:
                self.stage_times[self.current_stage] = []
                
            # Lưu thời gian
            self.stage_times[self.current_stage].append(elapsed)
            
            # Giới hạn kích thước
            if len(self.stage_times[self.current_stage]) > self.max_history:
                self.stage_times[self.current_stage].pop(0)
                
            self.current_stage = None
    
    def get_fps(self) -> float:
        """
        Tính tốc độ khung hình mỗi giây (FPS).
        
        Returns:
            float: Khung hình mỗi giây
        """
        with self._lock:
            elapsed_time = time.time() - self.start_time
            if elapsed_time == 0:
                return 0
            return self.frames_processed / elapsed_time
    
    def get_real_time_fps(self, window_size: int = 10) -> float:
        """
        Tính FPS thời gian thực dựa trên các khung hình gần đây nhất.
        
        Args:
            window_size: Số lượng khung hình cuối cùng để tính FPS
            
        Returns:
            float: FPS thời gian thực
        """
        with self._lock:
            if not self.inference_times:
                return 0
                
            # Lấy n mẫu cuối cùng
            recent_times = self.inference_times[-min(window_size, len(self.inference_times)):]
            if not recent_times:
                return 0
                
            # Tính trung bình thời gian xử lý
            avg_time = sum(recent_times) / len(recent_times)
            if avg_time == 0:
                return 0
                
            return 1.0 / avg_time
    
    def get_average_inference_time(self) -> float:
        """
        Tính thời gian suy luận trung bình cho mỗi khung hình.
        
        Returns:
            float: Thời gian trung bình (giây)
        """
        with self._lock:
            if self.frames_processed == 0:
                return 0
            return self.total_inference_time / self.frames_processed
    
    def get_inference_stats(self) -> Dict[str, float]:
        """
        Tính các thống kê về thời gian suy luận.
        
        Returns:
            Dict: Từ điển chứa các thống kê
                - 'mean': Thời gian trung bình
                - 'min': Thời gian tối thiểu
                - 'max': Thời gian tối đa
                - 'median': Thời gian trung vị
                - 'std_dev': Độ lệch chuẩn
        """
        with self._lock:
            if not self.inference_times:
                return {
                    'mean': 0,
                    'min': 0,
                    'max': 0,
                    'median': 0,
                    'std_dev': 0
                }
            
            return {
                'mean': statistics.mean(self.inference_times),
                'min': min(self.inference_times),
                'max': max(self.inference_times),
                'median': statistics.median(self.inference_times),
                'std_dev': statistics.stdev(self.inference_times) if len(self.inference_times) > 1 else 0
            }
    
    def get_stage_stats(self, stage_name: str) -> Dict[str, float]:
        """
        Tính các thống kê cho một giai đoạn cụ thể.
        
        Args:
            stage_name: Tên của giai đoạn cần lấy thống kê
            
        Returns:
            Dict: Thống kê cho giai đoạn đó (tương tự get_inference_stats)
        """
        with self._lock:
            if stage_name not in self.stage_times or not self.stage_times[stage_name]:
                return {
                    'mean': 0,
                    'min': 0,
                    'max': 0,
                    'median': 0,
                    'std_dev': 0
                }
            
            times = self.stage_times[stage_name]
            return {
                'mean': statistics.mean(times),
                'min': min(times),
                'max': max(times),
                'median': statistics.median(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Lấy tất cả thống kê, bao gồm tổng quát và theo từng giai đoạn.
        
        Returns:
            Dict: Từ điển chứa thống kê cho tất cả các giai đoạn và tổng quát
        """
        with self._lock:
            result = {
                'overall': self.get_inference_stats(),
                'fps': self.get_fps(),
                'real_time_fps': self.get_real_time_fps(),
                'total_frames': self.frames_processed,
                'uptime': time.time() - self.start_time
            }
            
            # Thêm thống kê cho từng giai đoạn
            for stage in self.stage_times:
                result[stage] = self.get_stage_stats(stage)
                
            return result
    
    def format_stats(self) -> str:
        """
        Định dạng thống kê thành chuỗi dễ đọc.
        
        Returns:
            str: Chuỗi chứa thông tin thống kê được định dạng
        """
        stats = self.get_all_stats()
        uptime = datetime.timedelta(seconds=int(stats['uptime']))
        
        result = [
            f"=== THỐNG KÊ HIỆU SUẤT ===",
            f"Thời gian hoạt động: {uptime}",
            f"Tổng số khung hình: {stats['total_frames']}",
            f"FPS (trung bình): {stats['fps']:.2f}",
            f"FPS (thời gian thực): {stats['real_time_fps']:.2f}",
            f"\nThời gian suy luận (ms):",
            f"  Trung bình: {stats['overall']['mean']*1000:.2f}",
            f"  Tối thiểu: {stats['overall']['min']*1000:.2f}",
            f"  Tối đa: {stats['overall']['max']*1000:.2f}",
            f"  Trung vị: {stats['overall']['median']*1000:.2f}",
            f"  Độ lệch chuẩn: {stats['overall']['std_dev']*1000:.2f}"
        ]
        
        # Thêm thông tin về các giai đoạn
        for stage, data in stats.items():
            if stage not in ['overall', 'fps', 'real_time_fps', 'total_frames', 'uptime']:
                result.append(f"\nGiai đoạn: {stage} (ms):")
                result.append(f"  Trung bình: {data['mean']*1000:.2f}")
                result.append(f"  Tối thiểu: {data['min']*1000:.2f}")
                result.append(f"  Tối đa: {data['max']*1000:.2f}")
                
        return "\n".join(result)
    
    def reset(self):
        """Đặt lại tất cả chỉ số."""
        with self._lock:
            self.start_time = time.time()
            self.frames_processed = 0
            self.total_inference_time = 0
            self.inference_times = []
            self.stage_times = {}
            self.current_stage = None


if __name__ == "__main__":
    # Ví dụ sử dụng
    monitor = PerformanceMonitor()
    
    # Giả lập xử lý 10 khung hình
    for i in range(10):
        # Bắt đầu một khung hình mới
        monitor.start_frame()
        
        # Giả lập giai đoạn phát hiện
        monitor.start_stage('detection')
        time.sleep(0.05)  # Giả lập 50ms cho phát hiện
        monitor.end_stage()
        
        # Giả lập giai đoạn ước tính độ sâu
        monitor.start_stage('depth')
        time.sleep(0.1)  # Giả lập 100ms cho độ sâu
        monitor.end_stage()
        
        # Kết thúc khung hình
        monitor.end_frame()
    
    # In thống kê 
    print(monitor.format_stats())
    
    # Lấy các thống kê lập trình
    stats = monitor.get_all_stats()
    print(f"\nFPS (Lập trình): {stats['fps']}")
    print(f"Thời gian trung bình giai đoạn phát hiện: {stats['detection']['mean']*1000:.2f}ms") 