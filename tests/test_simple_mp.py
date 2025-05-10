"""
Test đơn giản cho multiprocessing với 2 process: Camera và YOLO
"""
import multiprocessing as mp
import time
import numpy as np
import sys
import os

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MockCamera:
    """Camera giả lập đơn giản"""
    def __init__(self):
        self._n = 0
        self._size = (240, 320, 3)  # Kích thước nhỏ hơn để nhanh hơn
    
    def get_frame(self):
        self._n += 1
        frame = np.zeros(self._size, dtype=np.uint8)
        frame[50:100, 50:100, 0] = 255  # Vùng đỏ
        return frame

class MockYOLO:
    """YOLO giả lập đơn giản"""
    def __init__(self):
        self._n = 0
    
    def detect(self, frame):
        self._n += 1
        return {'boxes': [{'x': 50, 'y': 50, 'w': 50, 'h': 50}], 'frame_number': self._n}

def camera_process(frame_queue, run_event, ready_event):
    """Process con chụp ảnh từ camera"""
    try:
        print(f"[Camera P{os.getpid()}] Khởi động")
        camera = MockCamera()
        
        # Báo hiệu đã sẵn sàng
        ready_event.set()
        print(f"[Camera P{os.getpid()}] Đã sẵn sàng")
        
        count = 0
        while run_event.is_set():
            frame = camera.get_frame()
            frame_queue.put(frame)
            count += 1
            if count % 10 == 0:
                print(f"[Camera P{os.getpid()}] Đã xử lý {count} frames")
            time.sleep(0.01)  # Giảm tải CPU
    except Exception as e:
        print(f"[Camera P{os.getpid()}] Lỗi: {e}")
    finally:
        print(f"[Camera P{os.getpid()}] Kết thúc")

def yolo_process(frame_queue, detection_queue, run_event, ready_event):
    """Process con phát hiện đối tượng"""
    try:
        print(f"[YOLO P{os.getpid()}] Khởi động")
        yolo = MockYOLO()
        
        # Báo hiệu đã sẵn sàng
        ready_event.set()
        print(f"[YOLO P{os.getpid()}] Đã sẵn sàng")
        
        count = 0
        while run_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.5)
                if frame is not None:
                    result = yolo.detect(frame)
                    detection_queue.put(result)
                    count += 1
                    if count % 10 == 0:
                        print(f"[YOLO P{os.getpid()}] Đã xử lý {count} detections")
            except Exception as e:
                if "Empty" not in str(e):  # Bỏ qua lỗi queue rỗng
                    print(f"[YOLO P{os.getpid()}] Lỗi: {e}")
                time.sleep(0.01)  # Giảm tải CPU
    except Exception as e:
        print(f"[YOLO P{os.getpid()}] Lỗi: {e}")
    finally:
        print(f"[YOLO P{os.getpid()}] Kết thúc")

def main():
    """Chạy test đơn giản để kiểm tra multiprocessing"""
    # Đảm bảo sử dụng spawn cho Windows
    if mp.get_start_method() != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            print("Đã đặt spawn method ở nơi khác")
    
    print(f"Multiprocessing method: {mp.get_start_method()}")
    
    # Tạo các queue và event
    frame_queue = mp.Queue(maxsize=5)  # Queue nhỏ để giảm bộ nhớ
    detection_queue = mp.Queue(maxsize=5)
    run_event = mp.Event()
    camera_ready = mp.Event()
    yolo_ready = mp.Event()
    
    # Đặt run event
    run_event.set()
    
    # Khởi động các process
    camera_proc = mp.Process(
        target=camera_process,
        args=(frame_queue, run_event, camera_ready),
        daemon=True
    )
    
    yolo_proc = mp.Process(
        target=yolo_process,
        args=(frame_queue, detection_queue, run_event, yolo_ready),
        daemon=True
    )
    
    print("Đang khởi động các process...")
    camera_proc.start()
    yolo_proc.start()
    
    # Đợi các process sẵn sàng
    timeout = 5.0  # Giảm timeout xuống 5 giây
    print(f"Đang đợi các process sẵn sàng (timeout: {timeout}s)...")
    camera_ready_result = camera_ready.wait(timeout)
    yolo_ready_result = yolo_ready.wait(timeout)
    
    print(f"Camera ready: {camera_ready_result}")
    print(f"YOLO ready: {yolo_ready_result}")
    
    if not (camera_ready_result and yolo_ready_result):
        print("Một số process không sẵn sàng!")
        run_event.clear()
        time.sleep(1)
        return
    
    # Thu thập kết quả
    print("Đang thu thập kết quả...")
    results = []
    deadline = time.time() + 3  # Chỉ đợi 3 giây
    
    while time.time() < deadline and len(results) < 10:
        try:
            result = detection_queue.get(timeout=0.1)
            results.append(result)
            print(f"Nhận detection #{result['frame_number']}")
        except:
            pass
        time.sleep(0.05)
    
    # Dừng các process
    print(f"Đã nhận {len(results)} kết quả. Đang dừng các process...")
    run_event.clear()
    time.sleep(0.5)
    
    # Kiểm tra kết quả
    for i, result in enumerate(results):
        print(f"Result #{i+1}: frame_number={result['frame_number']}")
    
    print("Test hoàn thành.")

if __name__ == "__main__":
    main() 