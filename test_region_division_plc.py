"""
Test Region Division + PLC Integration
Demo chia pallets thành regions và gửi tọa độ vào PLC DB26

Chức năng test:
1. Test single image: Chia pallets và gửi vào PLC
2. Test real-time camera: Pipeline với PLC integration
3. Test PLC monitoring: Đọc giá trị từ PLC
"""
import cv2
import time
import os
import numpy as np
from typing import List, Dict, Any, Optional
from plc_communication import DB26Communication
from detection import YOLOTensorRT, ProcessingPipeline
from region_division_plc_integration import RegionDivisionPLCIntegration

def test_single_image_regions_plc():
    """
    Test 1: Chia pallets thành regions và gửi vào PLC với một ảnh đơn lẻ
    """
    print("=== TEST 1: SINGLE IMAGE REGIONS → PLC ===")
    print("Chia pallets thành 3 regions và gửi tọa độ Px, Py vào PLC DB26")
    print()
    
    # Cấu hình PLC
    plc_ip = input("Nhập IP PLC (mặc định: 192.168.0.1): ").strip()
    if not plc_ip:
        plc_ip = "192.168.0.1"
    
    try:
        plc_rack = input("Nhập Rack (mặc định: 0): ").strip()
        plc_rack = int(plc_rack) if plc_rack else 0
        
        plc_slot = input("Nhập Slot (mặc định: 1): ").strip() 
        plc_slot = int(plc_slot) if plc_slot else 1
    except ValueError:
        plc_rack = 0
        plc_slot = 1
    
    # Chọn layer
    layer_choice = input("Chọn layer (1 hoặc 2, mặc định 1): ").strip()
    try:
        layer = int(layer_choice) if layer_choice else 1
    except ValueError:
        layer = 1
    
    # Chọn ảnh test
    test_folders = ["images_pallets2", "images_pallets", "load_on_robot_images"]
    
    print("\nChọn folder ảnh:")
    for i, folder in enumerate(test_folders, 1):
        print(f"  {i}. {folder}")
    
    folder_choice = input(f"Chọn folder (1-{len(test_folders)}, mặc định 1): ").strip()
    try:
        folder_idx = int(folder_choice) - 1 if folder_choice else 0
        if 0 <= folder_idx < len(test_folders):
            pallets_folder = test_folders[folder_idx]
        else:
            pallets_folder = test_folders[0]
    except ValueError:
        pallets_folder = test_folders[0]
    
    if not os.path.exists(pallets_folder):
        print(f"❌ Không tìm thấy folder {pallets_folder}!")
        return
    
    image_files = [f for f in os.listdir(pallets_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    if not image_files:
        print(f"❌ Không có ảnh nào trong folder {pallets_folder}!")
        return
    
    print(f"\nẢnh có sẵn trong {pallets_folder}:")
    for i, img_file in enumerate(image_files, 1):
        print(f"  {i}. {img_file}")
    
    choice = input(f"\nChọn ảnh (1-{len(image_files)}, mặc định 1): ").strip()
    try:
        choice_num = int(choice) if choice else 1
        if 1 <= choice_num <= len(image_files):
            image_path = os.path.join(pallets_folder, image_files[choice_num - 1])
        else:
            image_path = os.path.join(pallets_folder, image_files[0])
    except ValueError:
        image_path = os.path.join(pallets_folder, image_files[0])
    
    # Đọc ảnh
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return
    
    print(f"\nXử lý ảnh: {os.path.basename(image_path)}")
    print(f"Layer: {layer}")
    print(f"PLC: {plc_ip}:{plc_rack}:{plc_slot}")
    
    # Khởi tạo system
    print(f"\nKhởi tạo YOLO model...")
    engine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.engine")
    yolo_model = YOLOTensorRT(engine_path=engine_path, conf=0.5)
    
    print(f"Khởi tạo Region Division PLC Integration...")
    region_plc = RegionDivisionPLCIntegration(
        plc_ip=plc_ip, 
        plc_rack=plc_rack, 
        plc_slot=plc_slot, 
        debug=True
    )
    
    # Kết nối PLC
    if not region_plc.connect_plc():
        print("❌ Không thể kết nối PLC! Tiếp tục demo mà không gửi dữ liệu...")
        show_demo = True
    else:
        show_demo = True
    
    try:
        # YOLO Detection
        print(f"\nThực hiện YOLO detection...")
        start_time = time.time()
        detections = yolo_model.detect(frame)
        yolo_time = time.time() - start_time
        
        print(f"✅ YOLO time: {yolo_time*1000:.2f} ms")
        print(f"✅ Phát hiện {len(detections.get('bounding_boxes', []))} objects")
        
        if len(detections.get('bounding_boxes', [])) == 0:
            print("❌ Không phát hiện object nào!")
            return
        
        # Hiển thị detected classes
        classes = detections.get('classes', [])
        class_names = {0: 'load', 1: 'load2', 2: 'pallet'}
        class_counts = {}
        for cls in classes:
            class_name = class_names.get(cls, f'unknown_{cls}')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"Classes detected: {class_counts}")
        
        # Region Division + PLC
        print(f"\nXử lý Region Division và gửi vào PLC...")
        start_time = time.time()
        regions_data, send_success = region_plc.process_detection_and_send_to_plc(detections, layer)
        process_time = time.time() - start_time
        
        print(f"✅ Region processing time: {process_time*1000:.2f} ms")
        print(f"✅ Đã chia được {len(regions_data)} regions")
        
        # Hiển thị chi tiết regions
        print(f"\nChi tiết regions:")
        for i, region_data in enumerate(regions_data):
            region_id = region_data['region_id']
            pallet_id = region_data['pallet_id']
            pixel_center = region_data['pixel_center']
            robot_coords = region_data['robot_coordinates']
            
            print(f"  Region {region_id} (Pallet {pallet_id}):")
            print(f"    Pixel center: ({pixel_center[0]:.1f}, {pixel_center[1]:.1f})")
            print(f"    Robot coords: Px={robot_coords['px']:.2f}, Py={robot_coords['py']:.2f}")
        
        # Đọc lại từ PLC để verify
        if region_plc.plc_connected:
            print(f"\nĐọc lại từ PLC để verify...")
            time.sleep(0.2)  # Đợi PLC xử lý
            plc_data = region_plc.read_regions_from_plc()
            
            print(f"Dữ liệu trong PLC DB26:")
            for region_name, data in plc_data.items():
                if data['px'] is not None and data['py'] is not None:
                    print(f"  {region_name}: Px={data['px']:7.2f} (DB26.{data['px_offset']}), "
                          f"Py={data['py']:7.2f} (DB26.{data['py_offset']})")
                else:
                    print(f"  {region_name}: ❌ Failed to read")
            
            if send_success:
                print(f"✅ PLC write/read test PASSED!")
            else:
                print(f"❌ PLC write/read test FAILED!")
        
        # Visualization
        if show_demo:
            print(f"\nTạo visualization...")
            vis_image = region_plc.create_visualization(frame, regions_data)
            
            # Hiển thị kết quả
            cv2.imshow("YOLO Detection", detections["annotated_frame"])
            cv2.imshow("Region Division + PLC Integration", vis_image)
            
            print(f"\nNhấn phím bất kỳ để tiếp tục...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Lưu kết quả
            save_choice = input("\nBạn có muốn lưu kết quả? (y/n): ").lower()
            if save_choice in ['y', 'yes']:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = f"test_region_plc_{base_name}_layer{layer}.jpg"
                cv2.imwrite(output_path, vis_image)
                print(f"✅ Đã lưu: {output_path}")
        
    finally:
        # Cleanup
        region_plc.disconnect_plc()

def test_realtime_camera_regions_plc():
    """
    Test 2: Real-time camera với Region Division + PLC Integration
    """
    print("\n=== TEST 2: REAL-TIME CAMERA REGIONS → PLC ===")
    print("Camera real-time với Region Division và gửi vào PLC")
    print()
    
    # Cấu hình PLC
    plc_ip = input("Nhập IP PLC (mặc định: 192.168.0.1): ").strip()
    if not plc_ip:
        plc_ip = "192.168.0.1"
    
    # Layer
    layer_choice = input("Chọn layer (1 hoặc 2, mặc định 1): ").strip()
    try:
        layer = int(layer_choice) if layer_choice else 1
    except ValueError:
        layer = 1
    
    print(f"\nSẽ chạy camera với:")
    print(f"  PLC IP: {plc_ip}")
    print(f"  Layer: {layer}")
    print(f"  Gửi tọa độ regions vào DB26")
    print()
    
    # Set environment variables cho pipeline
    os.environ['ENABLE_PLC_REGIONS'] = 'true'
    os.environ['PLC_IP'] = plc_ip
    os.environ['PLC_RACK'] = '0'
    os.environ['PLC_SLOT'] = '1'
    os.environ['SHOW_REGIONS'] = 'true'  # Hiển thị regions
    os.environ['DEPTH_DEVICE'] = 'off'  # Tắt depth để tối ưu
    
    print(f"Đã cấu hình environment variables:")
    print(f"  ENABLE_PLC_REGIONS=true")
    print(f"  PLC_IP={plc_ip}")
    print(f"  SHOW_REGIONS=true")
    print(f"  DEPTH_DEVICE=off")
    
    # Khởi tạo pipeline
    from use_tensorrt_example import create_camera, create_yolo, create_depth
    
    pipeline = ProcessingPipeline(
        camera_factory=create_camera,
        yolo_factory=create_yolo,
        depth_factory=create_depth
    )
    
    # Khởi động pipeline
    print(f"\nKhởi động pipeline...")
    if pipeline.start(timeout=60.0):
        print("✅ Pipeline đã khởi động thành công!")
        print("\nPhím điều khiển:")
        print("  'q': Thoát")
        print("  'r': Bật/tắt hiển thị regions")
        print("\nChú ý: Tọa độ regions sẽ được gửi vào PLC DB26 tự động")
        print("DB26 Layout: loads(0,4), pallets1(12,16), pallets2(24,28)")
        
        try:
            # Vòng lặp hiển thị kết quả
            fps_counter = 0
            fps_time = time.time()
            regions_sent_count = 0
            
            # Tracking để detect khi có regions mới
            last_regions_info = None
            
            while True:
                start_loop = time.time()
                
                # Lấy kết quả detection mới nhất
                detection_result = pipeline.get_latest_detection()
                if not detection_result:
                    time.sleep(0.01)
                    continue
                
                frame, detections = detection_result
                
                # Tính FPS
                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()
                
                # Kiểm tra pallet regions info
                pallet_regions = detections.get('pallet_regions', [])
                current_regions_info = None
                if pallet_regions:
                    current_regions_info = [(r['region_info']['region_id'], r['region_info']['pallet_id']) for r in pallet_regions]
                
                # Detect regions change
                if current_regions_info != last_regions_info:
                    if current_regions_info:
                        regions_sent_count += 1
                        print(f"\n[REGIONS UPDATE #{regions_sent_count}] Detected {len(pallet_regions)} regions:")
                        for region_data in pallet_regions:
                            center = region_data['center']
                            region_info = region_data['region_info']
                            print(f"  Region {region_info['region_id']} (Pallet {region_info['pallet_id']}): Center({center[0]:.1f}, {center[1]:.1f})")
                    last_regions_info = current_regions_info
                
                # Vẽ thông tin PLC trên frame
                display_frame = detections["annotated_frame"].copy()
                
                # Vẽ FPS
                fps_text = f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --"
                cv2.putText(display_frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Vẽ thông tin PLC
                plc_info = f"PLC: {plc_ip} | Regions sent: {regions_sent_count}"
                cv2.putText(display_frame, plc_info, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Vẽ số objects
                num_objects = len(detections.get('bounding_boxes', []))
                objects_text = f"Objects: {num_objects} | Regions: {len(pallet_regions)}"
                cv2.putText(display_frame, objects_text, (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Hiển thị main window
                cv2.imshow("Camera + Region Division + PLC", display_frame)
                
                # Xử lý phím nhấn
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Toggle regions display
                    show_regions = os.environ.get('SHOW_REGIONS', 'true')
                    new_show = 'false' if show_regions == 'true' else 'true'
                    os.environ['SHOW_REGIONS'] = new_show
                    print(f"Regions display: {new_show}")
        
        except KeyboardInterrupt:
            print("\n🛑 Đã nhận tín hiệu ngắt từ bàn phím")
        finally:
            # Dừng pipeline
            pipeline.stop()
            cv2.destroyAllWindows()
            print("✅ Pipeline đã dừng")
    else:
        print("❌ Không thể khởi động pipeline!")
        for error in pipeline.errors:
            print(f"Lỗi: {error}")

def test_plc_monitoring():
    """
    Test 3: Monitoring dữ liệu từ PLC DB26
    """
    print("\n=== TEST 3: PLC MONITORING ===")
    print("Đọc và monitoring tọa độ regions từ PLC DB26")
    print()
    
    # Cấu hình PLC
    plc_ip = input("Nhập IP PLC (mặc định: 192.168.0.1): ").strip()
    if not plc_ip:
        plc_ip = "192.168.0.1"
    
    # Tạo connection
    db26 = DB26Communication(plc_ip)
    
    if not db26.connect():
        print("❌ Không thể kết nối PLC!")
        return
    
    print("✅ Kết nối PLC thành công!")
    print("🔄 Bắt đầu monitoring (Nhấn Ctrl+C để dừng)...")
    print()
    
    print("DB26 Layout (Updated):")
    print("  loads: Px=DB26.0,  Py=DB26.4")
    print("  pallets1: Px=DB26.12, Py=DB26.16")
    print("  pallets2: Px=DB26.24, Py=DB26.28")
    print()
    
    try:
        last_values = {}
        
        while True:
            # Đọc giá trị hiện tại
            timestamp = time.strftime("%H:%M:%S")
            
            regions_data = {
                'loads': {'px_offset': 0, 'py_offset': 4},
                'pallets1': {'px_offset': 12, 'py_offset': 16},
                'pallets2': {'px_offset': 24, 'py_offset': 28}
            }
            
            current_values = {}
            changes_detected = False
            
            for region_name, offsets in regions_data.items():
                px = db26.read_db26_real(offsets['px_offset'])
                py = db26.read_db26_real(offsets['py_offset'])
                
                current_values[region_name] = {'px': px, 'py': py}
                
                # Detect changes
                if region_name not in last_values:
                    changes_detected = True
                elif (last_values[region_name]['px'] != px or 
                      last_values[region_name]['py'] != py):
                    changes_detected = True
            
            # Chỉ hiển thị khi có thay đổi hoặc mỗi 10 giây
            if changes_detected or int(time.time()) % 10 == 0:
                print(f"[{timestamp}] PLC DB26 Status:")
                for region_name, data in current_values.items():
                    px, py = data['px'], data['py']
                    if px is not None and py is not None:
                        status = "📍" if px != 0 or py != 0 else "⭕"
                        print(f"  {status} {region_name}: Px={px:8.2f}, Py={py:8.2f}")
                    else:
                        print(f"  ❌ {region_name}: Read error")
                print()
            
            last_values = current_values
            time.sleep(1)  # Đọc mỗi giây
    
    except KeyboardInterrupt:
        print("\n🛑 Dừng monitoring...")
    
    finally:
        db26.disconnect()
        print("✅ PLC đã ngắt kết nối.")

def main():
    """Menu chính"""
    print("🔧 REGION DIVISION + PLC INTEGRATION TEST TOOL")
    print("=" * 50)
    print("1. Test Single Image: Chia regions và gửi vào PLC")
    print("2. Test Real-time Camera: Pipeline với PLC integration") 
    print("3. Test PLC Monitoring: Đọc giá trị từ PLC")
    print("4. Chạy tất cả tests")
    
    choice = input("\nChọn test (1/2/3/4): ").strip()
    
    if choice == "1":
        test_single_image_regions_plc()
    elif choice == "2":
        test_realtime_camera_regions_plc()
    elif choice == "3":
        test_plc_monitoring()
    elif choice == "4":
        test_single_image_regions_plc()
        time.sleep(2)
        test_realtime_camera_regions_plc()
        time.sleep(2)
        test_plc_monitoring()
    else:
        print("❌ Lựa chọn không hợp lệ!")
        print("Chạy test mặc định...")
        test_single_image_regions_plc()

if __name__ == "__main__":
    main() 