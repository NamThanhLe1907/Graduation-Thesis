"""
Ví dụ sử dụng model TensorRT cho phát hiện đối tượng
"""
import cv2
import time
import os
import threading
from utils.detection import YOLOTensorRT
from utils.pipeline import ProcessingPipeline
from utils.camera import CameraInterface
from utils.detection.depth import DepthEstimator

# Đường dẫn tới file model - sử dụng .pt thay vì .engine để tránh lỗi version
ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "utils", "detection", "best.engine")
# Cấu hình hiển thị depth - mặc định là False để tránh lag
SHOW_DEPTH = os.environ.get('SHOW_DEPTH', 'false').lower() in ('true', '1', 'yes')

def demo_single_image():
    """Thử nghiệm với một ảnh đơn lẻ"""
    print("Thử nghiệm phát hiện đối tượng với TensorRT trên một ảnh đơn lẻ")
    
    # Khởi tạo model
    model = YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.25)
    
    # Đọc ảnh thử nghiệm
    image_path = input("Nhập đường dẫn tới ảnh thử nghiệm: ")
    if not image_path:
        image_path = "test.jpg"  # Ảnh mặc định
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return
    
    # Đo thời gian xử lý
    start_time = time.time()
    
    # Thực hiện phát hiện
    detections = model.detect(frame)
    
    end_time = time.time()
    
    # Hiển thị kết quả
    print(f"Thời gian xử lý: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Đã phát hiện {len(detections['bounding_boxes'])} đối tượng")
    
    # Hiển thị ảnh
    cv2.imshow("Kết quả phát hiện", detections["annotated_frame"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Di chuyển các hàm factory ra ngoài hàm demo_camera để có thể pickle
def create_camera():
    camera = CameraInterface(camera_index=0)
    camera.initialize()
    return camera

def create_yolo():
    return YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.25)

def create_depth():
    # Cho phép chạy depth model trên CPU hoặc tắt hoàn toàn
    use_device = os.environ.get('DEPTH_DEVICE', 'cuda')  # 'cuda', 'cpu' hoặc 'off' 
    enable_depth = use_device.lower() != 'off'
    
    # Lấy các tùy chọn cải thiện hiệu suất
    model_size = os.environ.get('DEPTH_MODEL', 'small')  # 'large', 'small', 'tiny'
    model_map = {
        'large': "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        'base': "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        'small': "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    }
    model_name = model_map.get(model_size.lower(), model_map['small'])
    
    # Kích thước input
    input_size_str = os.environ.get('DEPTH_SIZE', '640x640')
    input_size = None
    if input_size_str:
        try:
            w, h = map(int, input_size_str.split('x'))
            input_size = (w, h)
        except:
            print(f"[Factory] Không thể phân tích DEPTH_SIZE: {input_size_str}, sử dụng kích thước gốc")
    
    # Bỏ qua frame
    skip_frames_str = os.environ.get('DEPTH_SKIP', '20')
    try:
        skip_frames = int(skip_frames_str)
    except:
        skip_frames = 0
    
    if use_device.lower() == 'off':
        print(f"[Factory] Đã tắt depth model để tiết kiệm tài nguyên")
        return DepthEstimator(device='cpu', enable=False)
    
    print(f"[Factory] Khởi tạo depth model trên thiết bị: {use_device}")
    print(f"[Factory] Sử dụng model: {model_size} ({model_name})")
    
    return DepthEstimator(
        model_name=model_name,
        device=use_device, 
        enable=enable_depth,
        input_size=input_size,
        skip_frames=skip_frames
    )

def demo_camera():
    """Thử nghiệm với camera thời gian thực"""
    print("Thử nghiệm phát hiện đối tượng với TensorRT trên camera thời gian thực")
    global SHOW_DEPTH
    
    # Hiển thị tùy chọn depth
    print(f"Hiển thị depth map: {'BẬT' if SHOW_DEPTH else 'TẮT'} (Dùng 'd' để bật/tắt)")
    
    # Khởi tạo pipeline với các factory function đã được định nghĩa ở cấp module
    pipeline = ProcessingPipeline(
        camera_factory=create_camera,
        yolo_factory=create_yolo,
        depth_factory=create_depth
    )
    
    # Biến để lưu frame depth cuối cùng
    last_depth_viz = None
    last_depth_time = 0
    skip_counter = 0
    max_skip = 3  # Bỏ qua tối đa 3 frames khi xử lý không kịp
    
    # Khởi động pipeline
    if pipeline.start(timeout=60.0):
        print("Pipeline đã khởi động thành công!")
        
        try:
            # Vòng lặp hiển thị kết quả
            fps_counter = 0
            fps_time = time.time()
            
            while True:
                start_loop = time.time()
                
                # Lấy kết quả detection mới nhất
                detection_result = pipeline.get_latest_detection()
                if not detection_result:
                    # Nếu không có kết quả detection, chờ một chút
                    time.sleep(0.01)
                    continue
                
                frame, detections = detection_result
                
                # Nếu xử lý quá chậm, tăng skip_counter
                if time.time() - start_loop > 0.1:  # Quá 100ms
                    skip_counter += 1
                    if skip_counter >= max_skip:
                        # Bỏ qua hiển thị depth để giảm tải
                        skip_counter = 0
                        continue
                else:
                    skip_counter = 0  # Reset nếu xử lý nhanh
                
                # Tính FPS
                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()
                    
                    # Cập nhật thông tin thống kê
                    stats = pipeline.get_stats()
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Objects: {len(detections.get('bounding_boxes', []))}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Hiển thị kết quả detection - luôn hiển thị
                cv2.imshow("Phát hiện đối tượng với TensorRT", detections["annotated_frame"])
                
                # Xử lý depth chỉ khi SHOW_DEPTH được bật
                if SHOW_DEPTH:
                    # Chỉ lấy depth mới sau mỗi 0.5 giây
                    if time.time() - last_depth_time > 0.5:
                        depth_result = pipeline.get_latest_depth()
                        if depth_result:
                            frame_depth, depth_results = depth_result
                            
                            # Tạo bản sao một lần và chỉ xử lý đơn giản
                            depth_viz = frame_depth.copy()
                            
                            # Chỉ vẽ các bounding box đơn giản
                            for i, box_data in enumerate(depth_results):
                                bbox = box_data['bbox']
                                mean_depth = box_data['mean_depth']
                                x1, y1, x2, y2 = [int(v) for v in bbox]
                                
                                # Sử dụng màu đơn giản
                                cv2.rectangle(depth_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(depth_viz, f"{mean_depth:.1f}m", (x1, y1 - 5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            # Lưu lại để tái sử dụng
                            last_depth_viz = depth_viz
                            last_depth_time = time.time()
                    
                    # Hiển thị depth từ lần xử lý gần nhất
                    if last_depth_viz is not None:
                        cv2.imshow("Depth", last_depth_viz)
                
                # Xử lý phím nhấn
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    # Bật/tắt hiển thị depth
                    SHOW_DEPTH = not SHOW_DEPTH
                    print(f"Hiển thị depth map: {'BẬT' if SHOW_DEPTH else 'TẮT'}")
                    if not SHOW_DEPTH:
                        cv2.destroyWindow("Depth")
                        
        except KeyboardInterrupt:
            print("Đã nhận tín hiệu ngắt từ bàn phím")
        finally:
            # Dừng pipeline
            pipeline.stop()
            cv2.destroyAllWindows()
            print("Pipeline đã dừng")
    else:
        print("Không thể khởi động pipeline!")
        # Kiểm tra lỗi
        for error in pipeline.errors:
            print(f"Lỗi: {error}")

if __name__ == "__main__":
    print("Demo sử dụng model TensorRT")
    print("1. Thử nghiệm với ảnh đơn lẻ")
    print("2. Thử nghiệm với camera thời gian thực")
    print("\nGhi chú: Bạn có thể đặt các biến môi trường để điều khiển depth model:")
    print("  DEPTH_DEVICE: Thiết bị chạy depth model")
    print("    - DEPTH_DEVICE=cuda   # Chạy trên GPU (mặc định nếu có CUDA)")
    print("    - DEPTH_DEVICE=cpu    # Chạy trên CPU")
    print("    - DEPTH_DEVICE=off    # Tắt hoàn toàn depth model (mặc định)")
    print("\n  DEPTH_MODEL: Kích thước mô hình để tăng tốc độ")
    print("    - DEPTH_MODEL=large   # Mô hình lớn, chất lượng cao, chậm nhất")
    print("    - DEPTH_MODEL=base   # Mô hình vừa, cân bằng tốc độ/chất lượng")
    print("    - DEPTH_MODEL=small    # Mô hình nhỏ, tốc độ nhanh nhất")
    print("\n  DEPTH_SIZE: Kích thước đầu vào (W,H) để tăng tốc")
    print("    - DEPTH_SIZE=640x480  # Ví dụ: 640x480")
    print("\n  DEPTH_SKIP: Số frame bỏ qua giữa các lần xử lý")
    print("    - DEPTH_SKIP=5        # Ví dụ: Chỉ xử lý 1 frame trong mỗi 6 frames")
    print("\n  SHOW_DEPTH: Bật/tắt hiển thị depth map")
    print("    - SHOW_DEPTH=true     # Hiển thị depth map (có thể gây lag)")
    print("    - SHOW_DEPTH=false    # Tắt hiển thị depth map (mặc định)")
    print("\n  Ví dụ: set DEPTH_DEVICE=cpu && set DEPTH_MODEL=small && set DEPTH_SIZE=512x384 && set DEPTH_SKIP=5 && python use_tensorrt_example.py")
    print()
    
    choice = input("Chọn chế độ (1/2): ")
    
    if choice == "1":
        demo_single_image()
    elif choice == "2":
        demo_camera()
    else:
        print("Lựa chọn không hợp lệ!") 