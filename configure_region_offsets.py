"""
Script để cấu hình offset cho các regions trong RegionManager.
Sử dụng file này để thiết lập offset riêng cho từng region.
"""

import os
import sys

# Thêm đường dẫn để import detection modules
if '.' not in sys.path:
    sys.path.insert(0, '.')

from detection.utils.region_manager import RegionManager
import cv2
import numpy as np

def display_region_info(region_manager):
    """Hiển thị thông tin về tất cả regions"""
    print("\n" + "="*50)
    print("THÔNG TIN REGIONS HIỆN TẠI")
    print("="*50)
    
    for region_name, region_info in region_manager.regions.items():
        print(f"\n📍 {region_name.upper()}:")
        print(f"   Mô tả: {region_info['description']}")
        print(f"   Classes: {region_info['target_classes']}")
        print(f"   Polygon: {region_info['polygon']}")
        print(f"   Offset: X={region_info['offset']['x']}, Y={region_info['offset']['y']}")
        print(f"   Enabled: {region_info['enabled']}")
        print(f"   Priority: {region_info['priority']}")

def configure_offsets(region_manager):
    """Cấu hình offset cho từng region"""
    print("\n" + "="*50)
    print("CẤU HÌNH OFFSET CHO REGIONS")
    print("="*50)
    print("Nhập offset (X, Y) cho từng region.")
    print("Nhấn Enter để giữ nguyên giá trị hiện tại.")
    print("Nhập 'skip' để bỏ qua region.")
    
    for region_name in region_manager.regions.keys():
        current_offset = region_manager.regions[region_name]['offset']
        print(f"\n🔧 Cấu hình {region_name.upper()}:")
        print(f"   Offset hiện tại: X={current_offset['x']}, Y={current_offset['y']}")
        
        try:
            # Nhập offset X
            x_input = input(f"   Nhập offset X (hiện tại: {current_offset['x']}): ").strip()
            if x_input.lower() == 'skip':
                print(f"   ⏭️  Bỏ qua {region_name}")
                continue
            elif x_input == '':
                offset_x = current_offset['x']
            else:
                offset_x = float(x_input)
            
            # Nhập offset Y
            y_input = input(f"   Nhập offset Y (hiện tại: {current_offset['y']}): ").strip()
            if y_input == '':
                offset_y = current_offset['y']
            else:
                offset_y = float(y_input)
            
            # Áp dụng offset
            region_manager.set_region_offset(region_name, offset_x, offset_y)
            print(f"   ✅ Đã đặt offset cho {region_name}: X={offset_x}, Y={offset_y}")
            
        except ValueError:
            print(f"   ❌ Giá trị không hợp lệ, giữ nguyên offset cho {region_name}")
        except KeyboardInterrupt:
            print(f"\n   ⚠️  Dừng cấu hình offset")
            break
    
    # Hỏi có muốn save ra file không
    print(f"\n💾 Cấu hình offset hoàn tất!")
    save_choice = input("Bạn có muốn lưu cấu hình này ra file JSON? (y/n): ").strip().lower()
    if save_choice in ['y', 'yes']:
        file_name = input("Tên file (Enter để dùng 'region_offsets.json'): ").strip()
        if not file_name:
            file_name = "region_offsets.json"
        
        if region_manager.save_offsets_to_file(file_name):
            print(f"   ✅ Đã lưu cấu hình vào {file_name}")
        else:
            print(f"   ❌ Không thể lưu cấu hình vào {file_name}")

def test_region_detection(region_manager):
    """Test việc detection với một số điểm mẫu"""
    print("\n" + "="*50)
    print("TEST REGION DETECTION")
    print("="*50)
    
    # Một số điểm test mẫu
    test_points = [
        (500, 300, 0.0),   # Load trong vùng loads/pallets1
        (500, 300, 2.0),   # Pallet trong vùng loads/pallets1 
        (1000, 300, 2.0),  # Pallet trong vùng pallets2
        (1000, 300, 0.0),  # Load trong vùng pallets2 (sẽ không được assign)
        (100, 100, 1.0),   # Load ngoài vùng (sẽ unassigned)
    ]
    
    print("Test các điểm mẫu:")
    for i, (x, y, cls) in enumerate(test_points, 1):
        assigned_region = region_manager.get_region_for_detection((x, y), cls)
        class_names = {0.0: 'load', 1.0: 'load2', 2.0: 'pallet'}
        class_name = class_names.get(cls, f'class_{cls}')
        
        if assigned_region:
            print(f"   {i}. Point({x}, {y}) - {class_name} → 🎯 {assigned_region}")
        else:
            print(f"   {i}. Point({x}, {y}) - {class_name} → ❌ UNASSIGNED")

def visualize_regions():
    """Tạo ảnh minh họa các regions"""
    print("\n" + "="*50)
    print("TẠO ẢNH MINH HỌA REGIONS")
    print("="*50)
    
    # Tạo ảnh trắng với kích thước phù hợp
    img_width = 1280
    img_height = 720
    image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # Khởi tạo region manager
    region_manager = RegionManager()
    
    # Vẽ regions lên ảnh
    result_image = region_manager.draw_regions(image, show_labels=True)
    
    # Vẽ các điểm tọa độ gốc
    points_info = [
        ((821, 710), "1"),
        ((821, 3), "2"),
        ((2, 3), "3"),
        ((2, 710), "4"),
        ((1272, 710), "5"),
        ((1272, 3), "6"),
        ((356, 710), "7"),
        ((356, 3), "8"),
    ]
    
    for (x, y), label in points_info:
        # Vẽ điểm
        cv2.circle(result_image, (x, y), 8, (0, 0, 0), -1)
        cv2.circle(result_image, (x, y), 6, (255, 255, 255), -1)
        
        # Vẽ label
        cv2.putText(result_image, label, (x + 10, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Thêm legend
    legend_y = 30
    legend_items = [
        ("Loads (class 0,1): Points 1->2->8->7", (0, 255, 0)),
        ("Pallets1 (class 2): Points 1->2->8->7", (255, 0, 0)),
        ("Pallets2 (class 2): Points 1->2->6->5", (0, 0, 255)),
    ]
    
    for i, (text, color) in enumerate(legend_items):
        y_pos = legend_y + i * 25
        cv2.rectangle(result_image, (10, y_pos - 15), (30, y_pos - 5), color, -1)
        cv2.putText(result_image, text, (35, y_pos - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Lưu ảnh
    output_path = "regions_visualization.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"   ✅ Đã lưu ảnh minh họa tại: {output_path}")
    
    # Hiển thị ảnh nếu có thể
    try:
        cv2.imshow("Regions Visualization", result_image)
        print("   📺 Ảnh đã được hiển thị. Nhấn phím bất kỳ để đóng...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("   ⚠️  Không thể hiển thị ảnh (có thể chạy trên server)")

def load_offsets_from_file_interactive(region_manager):
    """Load offset từ file JSON với interaction"""
    print("\n" + "="*50)
    print("LOAD OFFSET TỪ FILE JSON")
    print("="*50)
    
    file_name = input("Nhập tên file JSON (Enter để dùng 'region_offsets.json'): ").strip()
    if not file_name:
        file_name = "region_offsets.json"
    
    print(f"📂 Đang load offset từ {file_name}...")
    
    if region_manager.load_offsets_from_file(file_name):
        print(f"   ✅ Load thành công từ {file_name}")
        print("\n📋 Offset sau khi load:")
        for region_name, region_info in region_manager.regions.items():
            offset = region_info['offset']
            print(f"   {region_name}: X={offset['x']}, Y={offset['y']}")
    else:
        print(f"   ❌ Không thể load từ {file_name}")

def save_offsets_to_file_interactive(region_manager):
    """Save offset ra file JSON với interaction"""
    print("\n" + "="*50)
    print("SAVE OFFSET RA FILE JSON")
    print("="*50)
    
    # Hiển thị offset hiện tại
    print("📋 Offset hiện tại:")
    for region_name, region_info in region_manager.regions.items():
        offset = region_info['offset']
        print(f"   {region_name}: X={offset['x']}, Y={offset['y']}")
    
    file_name = input("\nNhập tên file JSON để lưu (Enter để dùng 'region_offsets.json'): ").strip()
    if not file_name:
        file_name = "region_offsets.json"
    
    print(f"💾 Đang save offset ra {file_name}...")
    
    if region_manager.save_offsets_to_file(file_name):
        print(f"   ✅ Save thành công ra {file_name}")
    else:
        print(f"   ❌ Không thể save ra {file_name}")

def main():
    """Hàm chính"""
    print("🚀 CÔNG CỤ CẤU HÌNH REGION OFFSETS")
    print("Sử dụng tool này để thiết lập offset cho từng region.")
    
    # Khởi tạo RegionManager
    region_manager = RegionManager()
    
    while True:
        print("\n" + "="*50)
        print("MENU CHÍNH")
        print("="*50)
        print("1. Hiển thị thông tin regions hiện tại")
        print("2. Cấu hình offset cho regions")
        print("3. Test region detection với điểm mẫu")
        print("4. Tạo ảnh minh họa regions")
        print("5. Load offset từ file JSON")
        print("6. Save offset ra file JSON")
        print("7. Thoát")
        
        choice = input("\nChọn chức năng (1-7): ").strip()
        
        if choice == '1':
            display_region_info(region_manager)
        elif choice == '2':
            configure_offsets(region_manager)
        elif choice == '3':
            test_region_detection(region_manager)
        elif choice == '4':
            visualize_regions()
        elif choice == '5':
            load_offsets_from_file_interactive(region_manager)
        elif choice == '6':
            save_offsets_to_file_interactive(region_manager)
        elif choice == '7':
            print("👋 Thoát chương trình. Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main() 