"""
Ví dụ sử dụng PLC Communication với DB26
"""

from plc_communication import DB26Communication, PLCCommunication
import time

def example_basic_usage():
    """
    Ví dụ cơ bản sử dụng DB26Communication
    """
    print("=== Ví dụ cơ bản ===")
    
    # Tạo connection đến DB26
    db26 = DB26Communication(ip_address="192.168.0.1")
    
    try:
        # Kết nối
        if db26.connect():
            print("Kết nối thành công!")
            
            # Đọc một số giá trị
            print("\n--- Đọc dữ liệu từ DB26 ---")
            
            # Đọc BOOL tại DB26.0.0
            status = db26.read_db26_bool(0, 0)
            print(f"DB26.0.0 (Status): {status}")
            
            # Đọc INT tại DB26.2
            counter = db26.read_db26_int(2)
            print(f"DB26.2 (Counter): {counter}")
            
            # Đọc REAL tại DB26.8
            temperature = db26.read_db26_real(8)
            print(f"DB26.8 (Temperature): {temperature}")
            
            # Ghi dữ liệu (chỉ khi cần thiết)
            print("\n--- Ghi dữ liệu vào DB26 ---")
            
            # Ghi BOOL
            success = db26.write_db26_bool(1, 0, True)  # DB26.1.0 = True
            print(f"Ghi DB26.1.0 = True: {'Thành công' if success else 'Thất bại'}")
            
            # Ghi INT
            success = db26.write_db26_int(4, 100)  # DB26.4 = 100
            print(f"Ghi DB26.4 = 100: {'Thành công' if success else 'Thất bại'}")
            
        else:
            print("Không thể kết nối đến PLC")
            
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        db26.disconnect()

def example_monitoring_loop():
    """
    Ví dụ monitoring loop - đọc dữ liệu liên tục
    """
    print("\n=== Ví dụ Monitoring Loop ===")
    
    db26 = DB26Communication(ip_address="192.168.0.1")
    
    if not db26.connect():
        print("Không thể kết nối đến PLC")
        return
    
    try:
        print("Bắt đầu monitoring... (Nhấn Ctrl+C để dừng)")
        
        while True:
            # Đọc các giá trị quan trọng
            machine_running = db26.read_db26_bool(0, 0)  # DB26.0.0 - Machine Running
            cycle_count = db26.read_db26_int(2)          # DB26.2 - Cycle Count  
            speed = db26.read_db26_real(8)               # DB26.8 - Speed
            
            print(f"Machine Running: {machine_running}, Cycle: {cycle_count}, Speed: {speed:.2f}")
            
            time.sleep(1)  # Đọc mỗi giây
            
    except KeyboardInterrupt:
        print("\nDừng monitoring...")
    except Exception as e:
        print(f"Lỗi trong monitoring: {e}")
    finally:
        db26.disconnect()

def example_data_exchange():
    """
    Ví dụ trao đổi dữ liệu 2 chiều với PLC
    """
    print("\n=== Ví dụ Data Exchange ===")
    
    db26 = DB26Communication(ip_address="192.168.0.1")
    
    if not db26.connect():
        print("Không thể kết nối đến PLC")
        return
    
    try:
        # Định nghĩa data mapping cho DB26
        # (có thể customize theo nhu cầu thực tế)
        DATA_MAPPING = {
            # Inputs từ Python → PLC
            'python_ready': (10, 0),      # DB26.10.0 - Python ready signal
            'recipe_number': 12,           # DB26.12 - Recipe number (INT)
            'target_speed': 16,            # DB26.16 - Target speed (REAL)
            'command': 20,                 # DB26.20 - Command (DINT)
            
            # Outputs từ PLC → Python  
            'plc_ready': (0, 0),          # DB26.0.0 - PLC ready signal
            'current_position': 2,         # DB26.2 - Current position (INT)
            'actual_speed': 8,             # DB26.8 - Actual speed (REAL)
            'alarm_code': 24,              # DB26.24 - Alarm code (DINT)
        }
        
        # Gửi commands đến PLC
        print("Gửi commands đến PLC...")
        
        # Set Python ready
        db26.write_db26_bool(*DATA_MAPPING['python_ready'], True)
        
        # Set recipe number
        db26.write_db26_int(DATA_MAPPING['recipe_number'], 5)
        
        # Set target speed
        db26.write_db26_real(DATA_MAPPING['target_speed'], 1500.0)
        
        # Set command (1 = Start, 2 = Stop, 3 = Reset)
        db26.write_db26_dint(DATA_MAPPING['command'], 1)  # Start command
        
        # Đọc response từ PLC
        print("\nĐọc response từ PLC...")
        
        for i in range(10):  # Monitor trong 10 giây
            plc_ready = db26.read_db26_bool(*DATA_MAPPING['plc_ready'])
            position = db26.read_db26_int(DATA_MAPPING['current_position'])
            speed = db26.read_db26_real(DATA_MAPPING['actual_speed'])
            alarm = db26.read_db26_dint(DATA_MAPPING['alarm_code'])
            
            print(f"Loop {i+1}: PLC Ready: {plc_ready}, Position: {position}, Speed: {speed:.1f}, Alarm: {alarm}")
            
            time.sleep(1)
            
    except Exception as e:
        print(f"Lỗi trong data exchange: {e}")
    finally:
        db26.disconnect()

def example_custom_db():
    """
    Ví dụ sử dụng DB khác (không phải DB26)
    """
    print("\n=== Ví dụ Custom DB ===")
    
    # Sử dụng class PLCCommunication cơ bản để access DB khác
    plc = PLCCommunication(ip_address="192.168.0.1")
    
    if not plc.connect():
        print("Không thể kết nối đến PLC")
        return
    
    try:
        # Đọc từ DB1
        print("Đọc từ DB1...")
        db1_value = plc.read_int(1, 0)  # DB1.0
        print(f"DB1.0: {db1_value}")
        
        # Ghi vào DB2
        print("Ghi vào DB2...")
        success = plc.write_real(2, 0, 3.14159)  # DB2.0 = 3.14159
        print(f"Ghi DB2.0: {'Thành công' if success else 'Thất bại'}")
        
        # Đọc raw data từ DB26 (nếu cần)
        print("Đọc raw data từ DB26...")
        raw_data = plc.read_db_data(26, 0, 32)  # Đọc 32 bytes đầu của DB26
        if raw_data:
            print(f"Raw data: {raw_data.hex()}")
            
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        plc.disconnect()

if __name__ == "__main__":
    print("Chọn ví dụ để chạy:")
    print("1. Cơ bản")
    print("2. Monitoring Loop")
    print("3. Data Exchange") 
    print("4. Custom DB")
    print("5. Chạy tất cả")
    
    choice = input("Nhập lựa chọn (1-5): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_monitoring_loop()
    elif choice == "3":
        example_data_exchange()
    elif choice == "4":
        example_custom_db()
    elif choice == "5":
        example_basic_usage()
        example_data_exchange()
        example_custom_db()
        print("\nChạy monitoring loop cuối cùng...")
        example_monitoring_loop()
    else:
        print("Lựa chọn không hợp lệ!")
        print("Chạy ví dụ cơ bản...")
        example_basic_usage() 