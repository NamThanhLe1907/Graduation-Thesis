"""
Test script đơn giản cho PLC Integration
Gửi robot coordinates giả lập vào DB26 để test
"""
import time
import random
from plc_communication import DB26Communication

def test_plc_write_simulation():
    """
    Test ghi robot coordinates giả lập vào DB26
    """
    print("=== TEST PLC INTEGRATION ===")
    print("Script này sẽ gửi robot coordinates giả lập vào DB26")
    print("Px -> DB26.0 (offset 0)")
    print("Py -> DB26.4 (offset 4)")
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
        print("Sử dụng giá trị mặc định: Rack=0, Slot=1")
    
    # Tạo connection
    db26 = DB26Communication(plc_ip, plc_rack, plc_slot)
    
    if not db26.connect():
        print("❌ Không thể kết nối PLC!")
        return
    
    print("✅ Kết nối PLC thành công!")
    print()
    
    # Test 1: Ghi values cố định
    print("=== TEST 1: Ghi giá trị cố định ===")
    test_px = 100.0
    test_py = 200.0
    
    success_x = db26.write_db26_real(0, test_px)
    success_y = db26.write_db26_real(4, test_py)
    
    if success_x and success_y:
        print(f"✅ Đã ghi thành công: Px={test_px}, Py={test_py}")
    else:
        print(f"❌ Lỗi ghi dữ liệu")
    
    # Đọc lại để kiểm tra
    time.sleep(0.1)
    read_px = db26.read_db26_real(0)
    read_py = db26.read_db26_real(4)
    
    print(f"📖 Đọc lại từ PLC: Px={read_px}, Py={read_py}")
    print()
    
    # Test 2: Ghi values ngẫu nhiên liên tục
    print("=== TEST 2: Ghi giá trị ngẫu nhiên (10 lần) ===")
    
    for i in range(10):
        # Tạo coordinates ngẫu nhiên (giả lập robot workspace)
        px = round(random.uniform(-500, 500), 2)  # Ví dụ: -500mm đến +500mm
        py = round(random.uniform(-300, 300), 2)  # Ví dụ: -300mm đến +300mm
        pz = round(random.uniform(0, 100), 2)  # Ví dụ: 0mm đến 100mm
        # Ghi vào PLC
        success_x = db26.write_db26_real(0, px)
        success_y = db26.write_db26_real(4, py)
        success_z = db26.write_db26_real(8, pz)
        
        if success_x and success_y and success_z:
            print(f"✅ Lần {i+1}: Đã ghi Px={px:7.2f}, Py={py:7.2f}, Pz={pz:7.2f}")
        else:
            print(f"❌ Lần {i+1}: Lỗi ghi dữ liệu")
        
        time.sleep(1)  # Đợi 1 giây giữa các lần ghi
    
    print()
    
    # Test 3: Đọc giá trị cuối cùng
    print("=== TEST 3: Đọc giá trị cuối cùng ===")
    final_px = db26.read_db26_real(0)
    final_py = db26.read_db26_real(4)
    
    print(f"📖 Giá trị cuối cùng trong PLC:")
    print(f"   DB26.0 (Px): {final_px}")
    print(f"   DB26.4 (Py): {final_py}")
    
    # Cleanup
    db26.disconnect()
    print("\n✅ Test hoàn thành! PLC đã ngắt kết nối.")

def test_continuous_monitoring():
    """
    Test monitoring liên tục - đọc giá trị từ PLC
    """
    print("\n=== TEST CONTINUOUS MONITORING ===")
    print("Monitoring giá trị từ PLC (Nhấn Ctrl+C để dừng)")
    
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
    print("🔄 Bắt đầu monitoring...")
    print()
    
    try:
        while True:
            # Đọc giá trị hiện tại
            px = db26.read_db26_real(0)
            py = db26.read_db26_real(4)
            
            # Hiển thị với timestamp
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Px={px:8.2f}, Py={py:8.2f}")
            
            time.sleep(2)  # Đọc mỗi 2 giây
    
    except KeyboardInterrupt:
        print("\n🛑 Dừng monitoring...")
    
    finally:
        db26.disconnect()
        print("✅ PLC đã ngắt kết nối.")

def main():
    """Hàm main"""
    print("🔧 PLC Integration Test Tool")
    print("1. Test ghi robot coordinates vào DB26")
    print("2. Test monitoring liên tục từ DB26")
    print("3. Chạy cả hai test")
    
    choice = input("\nChọn test (1/2/3): ").strip()
    
    if choice == "1":
        test_plc_write_simulation()
    elif choice == "2":
        test_continuous_monitoring()
    elif choice == "3":
        test_plc_write_simulation()
        test_continuous_monitoring()
    else:
        print("❌ Lựa chọn không hợp lệ!")
        print("Chạy test mặc định...")
        test_plc_write_simulation()

if __name__ == "__main__":
    main() 