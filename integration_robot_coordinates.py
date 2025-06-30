"""
Integration script để truyền robot coordinates vào DB26 của PLC
Gửi Px, Py vào DB26 sau mỗi 100 frames với tùy chọn lưu dữ liệu
"""
import cv2
import time
import os
import threading
import json
from datetime import datetime
from typing import Optional, Dict, Any

from detection import (YOLOTensorRT, ProcessingPipeline, CameraInterface, DepthEstimator)
from plc_communication import DB26Communication

# Cấu hình đường dẫn model - tự động tìm model
def find_engine_path():
    """Tự động tìm đường dẫn model engine"""
    possible_paths = [
        "best.engine",  # Cùng thư mục
        os.path.join("models", "best.engine"),  # Thư mục models
        os.path.join("utils", "detection", "best.engine"),  # Từ use_tensorrt_example.py
        os.path.join("detection", "models", "best.engine"),  # Thư mục detection/models
        os.path.join("weights", "best.engine"),  # Thư mục weights
    ]
    
    for path in possible_paths:
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.exists(full_path):
            return full_path
    
    # Nếu không tìm thấy, return đường dẫn đầu tiên để user có thể điều chỉnh
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.engine")

ENGINE_PATH = find_engine_path()

class RobotCoordinatePLCIntegration:
    """
    Class integration để gửi robot coordinates vào PLC DB26
    """
    
    def __init__(self, plc_ip: str = "192.168.0.1", plc_rack: int = 0, plc_slot: int = 1):
        """
        Khởi tạo integration
        
        Args:
            plc_ip: IP address của PLC
            plc_rack: Rack number 
            plc_slot: Slot number
        """
        self.plc_ip = plc_ip
        self.plc_rack = plc_rack
        self.plc_slot = plc_slot
        
        # PLC communication
        self.db26 = DB26Communication(plc_ip, plc_rack, plc_slot)
        self.plc_connected = False
        
        # Frame counting
        self.frame_count = 0
        self.frames_per_update = 100  # Gửi PLC sau mỗi 100 frames
        
        # Data storage
        self.last_robot_coords = None
        self.save_data = False
        self.data_log = []
        
        # Pipeline
        self.pipeline = None
        
        # Stats
        self.total_plc_writes = 0
        self.successful_plc_writes = 0
        self.last_plc_write_time = None
        
    def connect_plc(self) -> bool:
        """
        Kết nối đến PLC
        
        Returns:
            bool: True nếu kết nối thành công
        """
        print(f"Đang kết nối đến PLC tại {self.plc_ip}...")
        self.plc_connected = self.db26.connect()
        
        if self.plc_connected:
            print("✓ Kết nối PLC thành công!")
        else:
            print("✗ Không thể kết nối PLC!")
        
        return self.plc_connected
    
    def disconnect_plc(self):
        """Ngắt kết nối PLC"""
        if self.plc_connected:
            self.db26.disconnect()
            self.plc_connected = False
            print("Đã ngắt kết nối PLC")
    
    def write_robot_coords_to_plc(self, px: float, py: float, pz: float = 0.0) -> bool:
        """
        Ghi robot coordinates vào DB26
        
        Args:
            px: Tọa độ X (ghi vào DB26.0 - offset 0)
            py: Tọa độ Y (ghi vào DB26.4 - offset 4) 
            pz: Tọa độ Z (ghi vào DB26.8 - offset 8, hiện tại không sử dụng)
            
        Returns:
            bool: True nếu ghi thành công
        """
        if not self.plc_connected:
            print("PLC chưa được kết nối!")
            return False
        
        try:
            # Ghi Px vào DB26.0 (offset 0)
            success_x = self.db26.write_db26_real(0, px)
            
            # Ghi Py vào DB26.4 (offset 4)
            success_y = self.db26.write_db26_real(4, py)
            
            # Ghi Pz vào DB26.8 (offset 8) - tạm thời comment out theo yêu cầu
            # success_z = self.db26.write_db26_real(8, pz)
            success_z = True  # Bỏ qua Pz hiện tại
            
            success = success_x and success_y and success_z
            
            if success:
                print(f"✓ Đã ghi robot coordinates vào PLC: X={px:.2f}, Y={py:.2f}")
                self.successful_plc_writes += 1
                self.last_plc_write_time = datetime.now()
            else:
                print(f"✗ Lỗi ghi robot coordinates vào PLC")
            
            self.total_plc_writes += 1
            return success
            
        except Exception as e:
            print(f"✗ Exception khi ghi PLC: {e}")
            self.total_plc_writes += 1
            return False
    
    def select_robot_coordinate(self, robot_coords: list) -> Optional[Dict]:
        """
        Chọn robot coordinate để gửi PLC (ưu tiên pallet, sau đó load)
        
        Args:
            robot_coords: Danh sách robot coordinates từ pipeline
            
        Returns:
            Dict hoặc None: Robot coordinate được chọn
        """
        if not robot_coords:
            return None
        
        # Ưu tiên 1: Pallet (class 2)
        pallets = [coord for coord in robot_coords if coord['class_id'] == 2]
        if pallets:
            return pallets[0]  # Lấy pallet đầu tiên
        
        # Ưu tiên 2: Load (class 0 hoặc 1)
        loads = [coord for coord in robot_coords if coord['class_id'] in [0, 1]]
        if loads:
            return loads[0]  # Lấy load đầu tiên
        
        # Fallback: Lấy object đầu tiên
        return robot_coords[0]
    
    def log_data(self, robot_coord: Dict, frame_count: int):
        """
        Log dữ liệu nếu được bật
        
        Args:
            robot_coord: Robot coordinate data
            frame_count: Số frame hiện tại
        """
        if not self.save_data:
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'frame_count': frame_count,
            'robot_coordinate': robot_coord,
            'plc_write_success': self.last_plc_write_time is not None
        }
        
        self.data_log.append(log_entry)
        
        # Giới hạn log size (chỉ giữ 1000 entries gần nhất)
        if len(self.data_log) > 1000:
            self.data_log = self.data_log[-1000:]
    
    def save_log_to_file(self):
        """Lưu log ra file JSON"""
        if not self.data_log:
            print("Không có dữ liệu để lưu")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robot_coords_plc_log_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.data_log, f, indent=2, ensure_ascii=False)
            print(f"✓ Đã lưu log vào {filename}")
        except Exception as e:
            print(f"✗ Lỗi lưu log: {e}")
    
    def create_camera_factory(self):
        """Factory function cho camera"""
        def create_camera():
            camera = CameraInterface(camera_index=0)
            camera.initialize()
            return camera
        return create_camera
    
    def create_yolo_factory(self):
        """Factory function cho YOLO"""
        def create_yolo():
            return YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.55)
        return create_yolo
    
    def create_depth_factory(self):
        """Factory function cho Depth (tắt để tiết kiệm tài nguyên)"""
        def create_depth():
            return DepthEstimator(device='cpu', enable=False)
        return create_depth
    
    def print_stats(self):
        """In thống kê"""
        success_rate = (self.successful_plc_writes / self.total_plc_writes * 100) if self.total_plc_writes > 0 else 0
        
        print(f"\n=== THỐNG KÊ PLC INTEGRATION ===")
        print(f"Tổng frames xử lý: {self.frame_count}")
        print(f"Tổng lần ghi PLC: {self.total_plc_writes}")
        print(f"Ghi PLC thành công: {self.successful_plc_writes}")
        print(f"Tỷ lệ thành công: {success_rate:.1f}%")
        
        if self.last_plc_write_time:
            print(f"Lần ghi cuối: {self.last_plc_write_time.strftime('%H:%M:%S')}")
        
        if self.save_data:
            print(f"Entries đã log: {len(self.data_log)}")
    
    def run(self, save_data: bool = False):
        """
        Chạy integration loop
        
        Args:
            save_data: Có lưu dữ liệu vào log không
        """
        self.save_data = save_data
        
        # Kết nối PLC
        if not self.connect_plc():
            print("Không thể tiếp tục do không kết nối được PLC")
            return
        
        # Khởi tạo pipeline
        print("Đang khởi tạo camera pipeline...")
        self.pipeline = ProcessingPipeline(
            camera_factory=self.create_camera_factory(),
            yolo_factory=self.create_yolo_factory(),
            depth_factory=self.create_depth_factory()
        )
        
        # Khởi động pipeline
        if not self.pipeline.start(timeout=60.0):
            print("Không thể khởi động camera pipeline!")
            self.disconnect_plc()
            return
        
        print("✓ Camera pipeline đã khởi động thành công!")
        print(f"Sẽ gửi robot coordinates vào PLC sau mỗi {self.frames_per_update} frames")
        print("Phím điều khiển:")
        print("  'q': Thoát")
        print("  's': Hiển thị thống kê")
        print("  'f': Thay đổi tần suất gửi PLC")
        if save_data:
            print("  'l': Lưu log ra file")
        print()
        
        try:
            fps_counter = 0
            fps_time = time.time()
            fps = 0.0
            
            while True:
                # Lấy kết quả detection
                detection_result = self.pipeline.get_latest_detection()
                if not detection_result:
                    time.sleep(0.01)
                    continue
                
                frame, detections = detection_result
                
                # Đếm frame
                self.frame_count += 1
                fps_counter += 1
                
                # Tính FPS
                if time.time() - fps_time >= 1.0:
                    fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()
                
                # Hiển thị frame với thông tin
                display_frame = detections["annotated_frame"].copy()
                
                # Vẽ thông tin lên frame
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.putText(display_frame, f"PLC Writes: {self.successful_plc_writes}/{self.total_plc_writes}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                frames_to_next = self.frames_per_update - (self.frame_count % self.frames_per_update)
                cv2.putText(display_frame, f"Next PLC: {frames_to_next}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                
                # Kiểm tra có robot coordinates không
                robot_coords = detections.get('robot_coordinates', [])
                if robot_coords:
                    cv2.putText(display_frame, f"Objects: {len(robot_coords)}", (10, 190), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Hiển thị coordinate sẽ được gửi
                    selected_coord = self.select_robot_coordinate(robot_coords)
                    if selected_coord:
                        robot_pos = selected_coord['robot_coordinates']
                        class_name = selected_coord['class']
                        cv2.putText(display_frame, f"Selected: {class_name} X={robot_pos['x']:.2f} Y={robot_pos['y']:.2f}", 
                                   (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow("Robot Coordinate PLC Integration", display_frame)
                
                # Gửi PLC sau mỗi N frames
                if self.frame_count % self.frames_per_update == 0 and robot_coords:
                    selected_coord = self.select_robot_coordinate(robot_coords)
                    if selected_coord:
                        robot_pos = selected_coord['robot_coordinates']
                        px = robot_pos['x']
                        py = robot_pos['y']
                        
                        # Ghi vào PLC
                        success = self.write_robot_coords_to_plc(px, py)
                        
                        # Lưu coordinate cuối cùng
                        self.last_robot_coords = selected_coord
                        
                        # Log dữ liệu
                        self.log_data(selected_coord, self.frame_count)
                        
                        print(f"Frame {self.frame_count}: Đã gửi {selected_coord['class']} coordinates đến PLC")
                
                # Xử lý phím nhấn
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.print_stats()
                elif key == ord('f'):
                    # Thay đổi tần suất gửi PLC
                    try:
                        new_freq = int(input(f"\nNhập tần suất mới (hiện tại: {self.frames_per_update}): "))
                        if new_freq > 0:
                            self.frames_per_update = new_freq
                            print(f"Đã thay đổi tần suất thành {self.frames_per_update} frames")
                        else:
                            print("Tần suất phải > 0")
                    except ValueError:
                        print("Giá trị không hợp lệ")
                elif key == ord('l') and save_data:
                    self.save_log_to_file()
        
        except KeyboardInterrupt:
            print("\nĐã nhận tín hiệu ngắt...")
        
        finally:
            # Cleanup
            if self.pipeline:
                self.pipeline.stop()
            
            cv2.destroyAllWindows()
            self.disconnect_plc()
            
            # In thống kê cuối
            self.print_stats()
            
            # Lưu log nếu được bật
            if save_data and self.data_log:
                save_choice = input("\nBạn có muốn lưu log ra file không? (y/n): ").lower()
                if save_choice in ['y', 'yes']:
                    self.save_log_to_file()

def main():
    """Hàm main"""
    print("=== Robot Coordinate PLC Integration ===")
    print("Script này sẽ:")
    print("1. Chạy camera pipeline để phát hiện objects")
    print("2. Tính toán robot coordinates (đã có offset)")
    print("3. Gửi coordinates vào DB26 của PLC sau mỗi N frames")
    print("4. Hiển thị real-time monitoring")
    print()
    print("Cấu hình DB26 PLC:")
    print("  - Px sẽ được ghi vào DB26.0 (offset 0, REAL)")
    print("  - Py sẽ được ghi vào DB26.4 (offset 4, REAL)")
    print("  - Pz hiện tại không được ghi (DB26.8)")
    print()
    
    # Cấu hình PLC
    print("Cấu hình PLC:")
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
        print("Sử dụng giá trị mặc định: Rack=0, Slot=1")
    
    # Tùy chọn lưu dữ liệu
    save_data_input = input("Bạn có muốn lưu dữ liệu vào log không? (y/n, mặc định: n): ").lower().strip()
    save_data = save_data_input in ['y', 'yes']
    
    # Tùy chọn tần suất gửi PLC
    try:
        freq_input = input("Nhập tần suất gửi PLC (số frames, mặc định: 100): ").strip()
        frames_per_update = int(freq_input) if freq_input else 100
    except ValueError:
        frames_per_update = 100
        print("Sử dụng tần suất mặc định: 100 frames")
    
    print(f"\nCấu hình:")
    print(f"  PLC: {plc_ip} (Rack: {plc_rack}, Slot: {plc_slot})")
    print(f"  Gửi PLC mỗi: {frames_per_update} frames")
    print(f"  Lưu log: {'Có' if save_data else 'Không'}")
    print(f"  Model path: {ENGINE_PATH}")
    print()
    
    # Kiểm tra file model có tồn tại không
    if not os.path.exists(ENGINE_PATH):
        print(f"⚠️  CẢNH BÁO: Không tìm thấy model tại {ENGINE_PATH}")
        print("Vui lòng điều chỉnh đường dẫn ENGINE_PATH trong file hoặc đặt model đúng vị trí")
        return
    
    # Tạo và chạy integration
    integration = RobotCoordinatePLCIntegration(plc_ip, plc_rack, plc_slot)
    integration.frames_per_update = frames_per_update
    
    try:
        integration.run(save_data=save_data)
    except Exception as e:
        print(f"Lỗi trong quá trình chạy: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 