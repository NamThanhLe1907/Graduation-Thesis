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
    
    # Khởi tạo model YOLO
    model = YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.25)
    
    # Khởi tạo model Depth (sử dụng chung config với camera)
    depth_model = create_depth()
    print(f"Depth model enable: {depth_model.enable}")
    
    # Hiển thị ảnh có sẵn từ folder images_pallets2
    print("\nẢnh có sẵn trong folder images_pallets2:")
    pallets_folder = "images_pallets2"
    if os.path.exists(pallets_folder):
        image_files = [f for f in os.listdir(pallets_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()  # Sắp xếp theo thứ tự
        for i, img_file in enumerate(image_files, 1):
            print(f"  {i}. {img_file}")
        print(f"  0. Nhập đường dẫn khác")
        
        choice = input(f"\nChọn ảnh (1-{len(image_files)}) hoặc 0 để nhập đường dẫn khác: ")
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(image_files):
                image_path = os.path.join(pallets_folder, image_files[choice_num - 1])
                print(f"Đã chọn: {image_path}")
            elif choice_num == 0:
                image_path = input("Nhập đường dẫn tới ảnh thử nghiệm: ")
                if not image_path:
                    image_path = "test.jpg"  # Ảnh mặc định
            else:
                print("Lựa chọn không hợp lệ, sử dụng ảnh đầu tiên")
                image_path = os.path.join(pallets_folder, image_files[0])
        except ValueError:
            print("Lựa chọn không hợp lệ, sử dụng ảnh đầu tiên")
            image_path = os.path.join(pallets_folder, image_files[0])
    else:
        # Đọc ảnh thử nghiệm theo cách cũ nếu không tìm thấy folder
        image_path = input("Nhập đường dẫn tới ảnh thử nghiệm: ")
        if not image_path:
            image_path = "test.jpg"  # Ảnh mặc định
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return
    
    # Hiển thị thông tin ảnh
    height, width = frame.shape[:2]
    print(f"\nThông tin ảnh:")
    print(f"  Đường dẫn: {image_path}")
    print(f"  Kích thước: {width}x{height}")
    print(f"  Kích thước file: {os.path.getsize(image_path)} bytes")
    
    # Đo thời gian xử lý YOLO
    start_time = time.time()
    
    # Thực hiện phát hiện YOLO
    detections = model.detect(frame)
    
    yolo_time = time.time()
    
    # Thực hiện depth estimation nếu được bật
    depth_results = None
    if depth_model.enable:
        print("Đang xử lý depth estimation...")
        depth_results = depth_model.estimate_depth(frame, detections['bounding_boxes'])
        
    depth_time = time.time()
    
    # Hiển thị kết quả
    print(f"Thời gian xử lý YOLO: {(yolo_time - start_time) * 1000:.2f} ms")
    if depth_model.enable:
        print(f"Thời gian xử lý Depth: {(depth_time - yolo_time) * 1000:.2f} ms")
        print(f"Tổng thời gian: {(depth_time - start_time) * 1000:.2f} ms")
    print(f"Đã phát hiện {len(detections['bounding_boxes'])} đối tượng")
    
    # Hiển thị thông tin depth nếu có
    if depth_results and len(depth_results) > 0:
        print("Thông tin độ sâu:")
        for i, result in enumerate(depth_results):
            print(f"  Đối tượng {i+1}: {result['mean_depth']:.2f}m (min: {result['min_depth']:.2f}m, max: {result['max_depth']:.2f}m)")
    
    # Hiển thị ảnh detection
    cv2.imshow("Kết quả phát hiện", detections["annotated_frame"])
    
    # Hiển thị depth map nếu có
    depth_viz = None
    if depth_model.enable and depth_results:
        # Tạo depth visualization
        depth_viz = frame.copy()
        for result in depth_results:
            bbox = result['bbox']
            mean_depth = result['mean_depth']
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Vẽ bounding box và thông tin depth
            cv2.rectangle(depth_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(depth_viz, f"{mean_depth:.1f}m", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Depth Information", depth_viz)
    
    # Lưu kết quả
    save_choice = input("\nBạn có muốn lưu kết quả? (y/n): ").lower()
    if save_choice in ['y', 'yes']:
        # Tạo tên file output
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        detection_output_path = f"result_{base_name}.jpg"
        cv2.imwrite(detection_output_path, detections["annotated_frame"])
        print(f"Đã lưu kết quả detection tại: {detection_output_path}")
        
        # Lưu depth visualization nếu có
        if depth_viz is not None:
            depth_output_path = f"depth_{base_name}.jpg"
            cv2.imwrite(depth_output_path, depth_viz)
            print(f"Đã lưu kết quả depth tại: {depth_output_path}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_batch_images():
    """Thử nghiệm với tất cả ảnh trong folder images_pallets2"""
    print("Thử nghiệm phát hiện đối tượng với TensorRT trên tất cả ảnh trong folder")
    
    pallets_folder = "images_pallets2"
    if not os.path.exists(pallets_folder):
        print(f"Không tìm thấy folder {pallets_folder}")
        return
    
    # Khởi tạo model YOLO
    model = YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.25)
    
    # Khởi tạo model Depth (sử dụng chung config với camera)
    depth_model = create_depth()
    print(f"Depth model enable: {depth_model.enable}")
    
    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(pallets_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    if not image_files:
        print("Không có ảnh nào trong folder")
        return
    
    print(f"Tìm thấy {len(image_files)} ảnh")
    
    # Tạo folder kết quả
    output_folder = "batch_results"
    os.makedirs(output_folder, exist_ok=True)
    
    # Tạo subfolder cho depth nếu depth được bật
    if depth_model.enable:
        depth_folder = os.path.join(output_folder, "depth")
        os.makedirs(depth_folder, exist_ok=True)
    
    total_time = 0
    total_yolo_time = 0
    total_depth_time = 0
    successful_detections = 0
    successful_depth_detections = 0
    
    for i, img_file in enumerate(image_files, 1):
        image_path = os.path.join(pallets_folder, img_file)
        print(f"\n[{i}/{len(image_files)}] Xử lý: {img_file}")
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"  Không thể đọc ảnh: {img_file}")
            continue
        
        # Đo thời gian xử lý YOLO
        start_time = time.time()
        
        # Thực hiện phát hiện YOLO
        detections = model.detect(frame)
        
        yolo_time = time.time()
        yolo_process_time = (yolo_time - start_time) * 1000
        total_yolo_time += yolo_process_time
        
        # Thực hiện depth estimation nếu được bật
        depth_results = None
        depth_process_time = 0
        if depth_model.enable:
            depth_results = depth_model.estimate_depth(frame, detections['bounding_boxes'])
            depth_end_time = time.time()
            depth_process_time = (depth_end_time - yolo_time) * 1000
            total_depth_time += depth_process_time
        
        total_process_time = yolo_process_time + depth_process_time
        total_time += total_process_time
        
        # Hiển thị kết quả
        num_objects = len(detections['bounding_boxes'])
        print(f"  Thời gian YOLO: {yolo_process_time:.2f} ms")
        if depth_model.enable:
            print(f"  Thời gian Depth: {depth_process_time:.2f} ms")
            print(f"  Tổng thời gian: {total_process_time:.2f} ms")
        print(f"  Đã phát hiện: {num_objects} đối tượng")
        
        if num_objects > 0:
            successful_detections += 1
        
        # Hiển thị thông tin depth nếu có
        if depth_results and len(depth_results) > 0:
            successful_depth_detections += 1
            print(f"  Thông tin độ sâu:")
            for j, result in enumerate(depth_results):
                print(f"    Đối tượng {j+1}: {result['mean_depth']:.2f}m")
        
        # Lưu kết quả detection
        base_name = os.path.splitext(img_file)[0]
        detection_output_path = os.path.join(output_folder, f"result_{base_name}.jpg")
        cv2.imwrite(detection_output_path, detections["annotated_frame"])
        print(f"  Đã lưu detection: {detection_output_path}")
        
        # Lưu kết quả depth nếu có
        if depth_model.enable and depth_results:
            # Tạo depth visualization
            depth_viz = frame.copy()
            for result in depth_results:
                bbox = result['bbox']
                mean_depth = result['mean_depth']
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # Vẽ bounding box và thông tin depth
                cv2.rectangle(depth_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(depth_viz, f"{mean_depth:.1f}m", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            depth_output_path = os.path.join(depth_folder, f"depth_{base_name}.jpg")
            cv2.imwrite(depth_output_path, depth_viz)
            print(f"  Đã lưu depth: {depth_output_path}")
    
    # Thống kê tổng kết
    print(f"\n=== THỐNG KÊ TỔNG KẾT ===")
    print(f"Tổng số ảnh xử lý: {len(image_files)}")
    print(f"Ảnh có phát hiện đối tượng: {successful_detections}")
    print(f"Tỉ lệ phát hiện thành công: {successful_detections/len(image_files)*100:.1f}%")
    
    if depth_model.enable:
        print(f"Ảnh có thông tin depth: {successful_depth_detections}")
        print(f"Tỉ lệ depth thành công: {successful_depth_detections/len(image_files)*100:.1f}%")
        print(f"Thời gian YOLO trung bình: {total_yolo_time/len(image_files):.2f} ms/ảnh")
        print(f"Thời gian Depth trung bình: {total_depth_time/len(image_files):.2f} ms/ảnh")
    
    print(f"Thời gian tổng trung bình: {total_time/len(image_files):.2f} ms/ảnh")
    print(f"Kết quả detection đã được lưu trong folder: {output_folder}")
    
    if depth_model.enable:
        print(f"Kết quả depth đã được lưu trong folder: {os.path.join(output_folder, 'depth')}")

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
    
    # Lấy loại model: regular hoặc metric
    model_type = os.environ.get('DEPTH_TYPE', 'metric').lower()  # 'regular' hoặc 'metric'
    
    # Lấy kích thước model
    model_size = os.environ.get('DEPTH_MODEL', 'small').lower()  # 'large', 'base', 'small'
    
    # Lấy loại scene cho metric depth
    scene_type = os.environ.get('DEPTH_SCENE', 'indoor').lower()  # 'indoor' hoặc 'outdoor'
    
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
    print(f"[Factory] Model type: {model_type}, Size: {model_size}")
    if model_type == 'metric':
        print(f"[Factory] Scene type: {scene_type}")
    
    # Tạo DepthEstimator dựa trên loại model
    if model_type == 'metric':
        return DepthEstimator.create_metric(
            scene_type=scene_type,
            model_size=model_size,
            device=use_device, 
            enable=enable_depth,
            input_size=input_size,
            skip_frames=skip_frames
        )
    else:
        return DepthEstimator.create_regular(
            model_size=model_size,
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
    print("1. Thử nghiệm với ảnh đơn lẻ (có depth estimation)")
    print("2. Thử nghiệm với camera thời gian thực (có depth estimation)")
    print("3. Thử nghiệm với tất cả ảnh trong folder images_pallets2 (có depth estimation)")
    print("\nGhi chú: Tất cả các demo đều sử dụng chung cấu hình depth model!")
    print("Bạn có thể đặt các biến môi trường để điều khiển depth model:")
    print("  DEPTH_DEVICE: Thiết bị chạy depth model")
    print("    - DEPTH_DEVICE=cuda   # Chạy trên GPU (mặc định nếu có CUDA)")
    print("    - DEPTH_DEVICE=cpu    # Chạy trên CPU")
    print("    - DEPTH_DEVICE=off    # Tắt hoàn toàn depth model (mặc định)")
    print("\n  DEPTH_TYPE: Loại mô hình depth")
    print("    - DEPTH_TYPE=regular  # Regular depth model (normalized output)")
    print("    - DEPTH_TYPE=metric   # Metric depth model (output in meters)")
    print("\n  DEPTH_MODEL: Kích thước mô hình để tăng tốc độ")
    print("    - DEPTH_MODEL=large   # Mô hình lớn, chất lượng cao, chậm nhất")
    print("    - DEPTH_MODEL=base    # Mô hình vừa, cân bằng tốc độ/chất lượng")
    print("    - DEPTH_MODEL=small   # Mô hình nhỏ, tốc độ nhanh nhất (mặc định)")
    print("\n  DEPTH_SCENE: Loại cảnh (chỉ cho metric depth)")
    print("    - DEPTH_SCENE=indoor  # Cảnh trong nhà (mặc định)")
    print("    - DEPTH_SCENE=outdoor # Cảnh ngoài trời")
    print("\n  DEPTH_SIZE: Kích thước đầu vào (W,H) để tăng tốc")
    print("    - DEPTH_SIZE=640x480  # Ví dụ: 640x480")
    print("\n  DEPTH_SKIP: Số frame bỏ qua giữa các lần xử lý")
    print("    - DEPTH_SKIP=5        # Ví dụ: Chỉ xử lý 1 frame trong mỗi 6 frames")
    print("\n  SHOW_DEPTH: Bật/tắt hiển thị depth map")
    print("    - SHOW_DEPTH=true     # Hiển thị depth map (có thể gây lag)")
    print("    - SHOW_DEPTH=false    # Tắt hiển thị depth map (mặc định)")
    print("\n  Ví dụ Regular Depth:")
    print("    set DEPTH_DEVICE=cuda && set DEPTH_TYPE=regular && set DEPTH_MODEL=small && python use_tensorrt_example.py")
    print("\n  Ví dụ Metric Depth (Indoor):")
    print("    set DEPTH_DEVICE=cuda && set DEPTH_TYPE=metric && set DEPTH_SCENE=indoor && set DEPTH_MODEL=base && python use_tensorrt_example.py")
    print("\n  Ví dụ Metric Depth (Outdoor):")
    print("    set DEPTH_DEVICE=cuda && set DEPTH_TYPE=metric && set DEPTH_SCENE=outdoor && set DEPTH_MODEL=small && python use_tensorrt_example.py")
    print()
    
    choice = input("Chọn chế độ (1/2/3): ")
    
    if choice == "1":
        demo_single_image()
    elif choice == "2":
        demo_camera()
    elif choice == "3":
        demo_batch_images()
    else:
        print("Lựa chọn không hợp lệ!") 