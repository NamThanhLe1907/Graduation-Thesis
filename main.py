import cv2
import tkinter as tk
from threading import Thread
from utils.camera_interface import CameraInterface
from utils.frame_processor import FrameProcessor
from utils.yolo_inference import YOLOInference
from utils.postprocessing import PostProcessor
from utils.visualization import AppGUI
from utils.performance_monitor import PerformanceMonitor
import time

class VideoProcessor:
    def __init__(self, gui):
        self.gui = gui
        self.camera = CameraInterface()
        self.processor = FrameProcessor()
        self.yolo_inference = YOLOInference(model_path="best.pt")  # Đảm bảo model path đúng
        self.post_processor = PostProcessor()
        self.performance_monitor = PerformanceMonitor()
        self.running = False

    def start_processing(self):
        self.running = True
        self.camera.initialize()
        
        while self.running:
            try:
                self.performance_monitor.start_frame()
                
                # Capture frame từ camera
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                
                # Preprocessing: xử lý frame gốc trước khi inference
                processed_frame = self.processor.preprocess_frame(frame)
                
                # Khởi tạo annotated_frame từ processed_frame ban đầu
                annotated_frame = processed_frame.copy()
                
                # Chạy YOLO inference
                results = self.yolo_inference.infer(processed_frame)
                self.gui.log_message(f"YOLO results: {len(results)} detections", "DEBUG")
                if results and len(results) > 0:
                    try:
                        # Giả sử results[0].plot() trả về frame annotated
                        annotated_frame = results[0].plot()
                        diff = cv2.norm(annotated_frame, processed_frame)
                        self.gui.log_message(f"Annotation applied successfully, diff={diff}, shape={annotated_frame.shape}", "DEBUG")
                    except Exception as e:
                        self.gui.log_message(f"Error in result plot: {e}", "ERROR")
                        annotated_frame = processed_frame.copy()
                else:
                    self.gui.log_message("No inference results", "WARNING")
       
                # Debug: lưu file hình annotated để kiểm tra
                cv2.imwrite("debug_output.jpg", annotated_frame)
                
                # Sử dụng annotated_frame (nếu có 4 kênh thì chuyển đổi sang BGR)
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGRA2BGR) if (len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 4) else annotated_frame
                self.gui.root.after(0, lambda: self.gui.update(display_frame))
                
                self.performance_monitor.end_frame()
                time.sleep(0.01)
                
            except Exception as e:
                self.gui.log_message(f"Processing error: {str(e)}", "ERROR")
                break

        self.camera.release()
        self.gui.log_message("Camera released")

def main():
    root = tk.Tk()
    gui = AppGUI(root)
    processor = VideoProcessor(gui)
    
    # Khởi chạy xử lý video trên 1 thread riêng
    processing_thread = Thread(target=processor.start_processing)
    processing_thread.daemon = True
    processing_thread.start()
    
    print("Ứng dụng đang khởi động...")
    gui.run()
    print("Ứng dụng đã tắt")

if __name__ == "__main__":
    print("Bắt đầu chạy chương trình từ main")
    main()
