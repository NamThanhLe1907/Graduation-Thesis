"""
Demo cụ thể cho pallet hình chữ nhật 12x10cm.
Giải thích rõ ràng hệ quy chiếu góc xoay cho hình chữ nhật.
"""
import cv2
import numpy as np
import math
import os

def draw_rectangle_at_angle(image, center, width, height, angle_deg, color, thickness=2):
    """
    Vẽ hình chữ nhật xoay tại góc nhất định.
    
    Args:
        image: Ảnh để vẽ lên
        center: Tâm hình chữ nhật (cx, cy)
        width: Chiều rộng (cạnh theo trục X khi góc = 0°)
        height: Chiều cao (cạnh theo trục Y khi góc = 0°)
        angle_deg: Góc xoay (độ)
        color: Màu sắc
        thickness: Độ dày viền
    """
    cx, cy = center
    
    # Tính 4 góc của hình chữ nhật
    # Góc khi chưa xoay (ở tâm gốc tọa độ)
    corners_local = np.array([
        [-width/2, -height/2],  # Top-left
        [width/2, -height/2],   # Top-right  
        [width/2, height/2],    # Bottom-right
        [-width/2, height/2]    # Bottom-left
    ])
    
    # Ma trận xoay
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Xoay các góc
    corners_rotated = corners_local @ rotation_matrix.T
    
    # Di chuyển về vị trí center
    corners_world = corners_rotated + np.array([cx, cy])
    
    # Vẽ hình chữ nhật
    pts = corners_world.astype(np.int32)
    cv2.fillPoly(image, [pts], color)
    cv2.polylines(image, [pts], True, (0, 0, 0), thickness)
    
    return corners_world

def draw_pallet_with_details(image, center, angle_deg, pallet_size=(120, 100), label=""):
    """
    Vẽ pallet với thông tin chi tiết.
    
    Args:
        image: Ảnh để vẽ lên
        center: Tâm pallet
        angle_deg: Góc xoay
        pallet_size: Kích thước pallet (width, height) theo pixels (tỷ lệ 12:10)
        label: Nhãn mô tả
    """
    cx, cy = center
    width, height = pallet_size
    
    # Màu pallet
    pallet_color = (200, 150, 100)  # Màu gỗ
    
    # Vẽ pallet
    corners = draw_rectangle_at_angle(image, center, width, height, angle_deg, pallet_color, 3)
    
    # Vẽ tâm
    cv2.circle(image, (int(cx), int(cy)), 5, (255, 255, 255), -1)
    cv2.circle(image, (int(cx), int(cy)), 8, (0, 0, 0), 2)
    
    # Vẽ trục dài (major axis) - màu đỏ
    major_length = max(width, height) * 0.4
    major_end_x = cx + major_length * math.cos(math.radians(angle_deg))
    major_end_y = cy + major_length * math.sin(math.radians(angle_deg))
    cv2.arrowedLine(image, (int(cx), int(cy)), (int(major_end_x), int(major_end_y)), 
                   (0, 0, 255), 3, tipLength=0.3)
    
    # Vẽ trục ngắn (minor axis) - màu xanh lá
    minor_angle = angle_deg + 90
    minor_length = min(width, height) * 0.4
    minor_end_x = cx + minor_length * math.cos(math.radians(minor_angle))
    minor_end_y = cy + minor_length * math.sin(math.radians(minor_angle))
    cv2.arrowedLine(image, (int(cx), int(cy)), (int(minor_end_x), int(minor_end_y)), 
                   (0, 255, 0), 3, tipLength=0.3)
    
    # Vẽ cung góc
    if angle_deg != 0:
        radius = 40
        start_angle = 0
        end_angle = int(angle_deg)
        
        if angle_deg > 0:
            cv2.ellipse(image, (int(cx), int(cy)), (radius, radius), 0, 
                       start_angle, end_angle, (255, 255, 0), 2)
        else:
            cv2.ellipse(image, (int(cx), int(cy)), (radius, radius), 0, 
                       end_angle, start_angle, (255, 255, 0), 2)
    
    # Text thông tin
    text_lines = [
        f"{label}",
        f"Goc: {angle_deg}°",
        f"12cm x 10cm"
    ]
    
    # Vẽ text với background
    y_start = int(cy) - 80
    for i, line in enumerate(text_lines):
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        y_pos = y_start + i * 25
        
        # Background
        cv2.rectangle(image, 
                     (int(cx) - text_size[0]//2 - 5, y_pos - text_size[1] - 5),
                     (int(cx) + text_size[0]//2 + 5, y_pos + 5),
                     (255, 255, 255), -1)
        
        # Text
        cv2.putText(image, line, (int(cx) - text_size[0]//2, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Chú thích trục
    cv2.putText(image, "Truc dai (12cm)", (int(major_end_x) + 10, int(major_end_y)), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(image, "Truc ngan (10cm)", (int(minor_end_x) + 10, int(minor_end_y)), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

def create_pallet_orientation_guide():
    """
    Tạo ảnh hướng dẫn orientation cho pallet 12x10cm.
    """
    # Tạo ảnh lớn để chứa nhiều ví dụ
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 240
    
    # Title
    cv2.putText(img, "HUONG DAN GOC XOAY CHO PALLET 12x10cm", 
               (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Subtitle
    cv2.putText(img, "Canh dai 12cm (do), Canh ngan 10cm (xanh)", 
               (300, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Vẽ hệ tọa độ chung
    cv2.putText(img, "He toa do: X+ sang phai, Y+ xuong duoi", 
               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Các vị trí để vẽ pallet
    positions = [
        (200, 200, 0, "0° - NAM NGANG\n(Canh dai theo X+)"),
        (500, 200, 30, "30° - NGHIENG NHE"),
        (800, 200, 45, "45° - NGHIENG 45°"),
        (1000, 200, 90, "90° - DUNG THANG\n(Canh dai theo Y+)"),
        
        (200, 450, 125, "125° - NGHIENG XUONG"),
        (500, 450, 180, "180° - NAM NGANG NGUOC\n(Canh dai theo X-)"),
        (800, 450, -90, "-90° - DUNG THANG NGUOC\n(Canh dai theo Y-)"),
        (1000, 450, -45, "-45° - NGHIENG LEN"),
    ]
    
    for x, y, angle, description in positions:
        draw_pallet_with_details(img, (x, y), angle, (120, 100), description)
    
    # Thêm chú thích quan trọng
    notes = [
        "QUAN TRONG:",
        "- Goc 0°: Canh dai (12cm) nam ngang theo X+ (huong phai)",
        "- Goc 90°: Canh dai (12cm) thang dung theo Y+ (huong xuong)",
        "- Goc duong (+): Xoay nguoc chieu kim dong ho",
        "- Goc am (-): Xoay cung chieu kim dong ho",
        "",
        "CHU Y YOLO OBB:",
        "- YOLO co the dinh nghia goc khac nhau tuy theo training data",
        "- Can kiem tra bang demo tren anh that"
    ]
    
    y_start = 600
    for i, note in enumerate(notes):
        if note == "":
            continue
        color = (0, 0, 255) if note.startswith("QUAN TRONG") or note.startswith("CHU Y") else (0, 0, 0)
        weight = 2 if note.startswith("-") else 1
        cv2.putText(img, note, (50, y_start + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, weight)
    
    return img

def create_yolo_vs_standard_comparison():
    """
    So sánh giữa định nghĩa góc chuẩn và YOLO có thể khác biệt.
    """
    img = np.ones((600, 1000, 3), dtype=np.uint8) * 240
    
    # Title
    cv2.putText(img, "SO SANH: DINH NGHIA GOC CHUAN vs YOLO THUC TE", 
               (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Cột 1: Định nghĩa chuẩn
    cv2.putText(img, "DINH NGHIA CHUAN", (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "(Theo ly thuyet Computer Vision)", (100, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    standard_positions = [
        (200, 150, 0, "0°"),
        (200, 230, 90, "90°"),
        (200, 310, 180, "180°"),
        (200, 390, -90, "-90°")
    ]
    
    for x, y, angle, label in standard_positions:
        draw_pallet_with_details(img, (x, y), angle, (80, 60), label)
    
    # Cột 2: YOLO thực tế (có thể khác)
    cv2.putText(img, "YOLO THUC TE", (600, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, "(Can kiem tra bang demo)", (550, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.putText(img, "Chay rotation_reference_demo.py", (500, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, "de xem YOLO dinh nghia nhu the nao!", (480, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Vẽ dấu hỏi cho YOLO
    yolo_positions = [
        (650, 200, "?"),
        (650, 280, "?"),
        (650, 360, "?"),
        (650, 440, "?")
    ]
    
    for x, y, label in yolo_positions:
        cv2.circle(img, (x, y), 40, (200, 200, 200), -1)
        cv2.circle(img, (x, y), 40, (0, 0, 0), 2)
        cv2.putText(img, label, (x-10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # Chú thích
    notes = [
        "YOLO co the:",
        "- Dinh nghia goc 0° la canh ngan thay vi canh dai",
        "- Su dung he toa do khac",
        "- Phu thuoc vao cach train model",
        "",
        "=> CAN KIEM TRA BANG ANH THAT!"
    ]
    
    for i, note in enumerate(notes):
        if note == "":
            continue
        cv2.putText(img, note, (50, 500 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img

def demo_pallet_orientation():
    """
    Demo orientation cho pallet 12x10cm.
    """
    print("=== DEMO ORIENTATION CHO PALLET 12x10cm ===")
    print()
    print("Pallet hình chữ nhật:")
    print("- Cạnh dài: 12cm")
    print("- Cạnh ngắn: 10cm")
    print()
    print("Sẽ hiển thị các góc xoay khác nhau...")
    
    # Tạo và hiển thị ảnh hướng dẫn
    orientation_guide = create_pallet_orientation_guide()
    cv2.imshow("Huong Dan Goc Xoay Pallet 12x10cm", orientation_guide)
    
    print("\nHiển thị hướng dẫn góc xoay. Nhấn phím bất kỳ để tiếp tục...")
    cv2.waitKey(0)
    
    # Hiển thị so sánh
    comparison = create_yolo_vs_standard_comparison()
    cv2.imshow("So Sanh Dinh Nghia Goc", comparison)
    
    print("\nHiển thị so sánh định nghĩa góc. Nhấn phím bất kỳ để tiếp tục...")
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    # Lưu ảnh
    save_choice = input("\nBạn có muốn lưu các ảnh hướng dẫn? (y/n): ").lower()
    if save_choice in ['y', 'yes']:
        cv2.imwrite("pallet_orientation_guide.jpg", orientation_guide)
        cv2.imwrite("yolo_vs_standard_comparison.jpg", comparison)
        print("Đã lưu:")
        print("  - pallet_orientation_guide.jpg")
        print("  - yolo_vs_standard_comparison.jpg")
    
    print("\n" + "="*60)
    print("KẾT LUẬN:")
    print("="*60)
    print("Theo lý thuyết Computer Vision chuẩn:")
    print("• Góc 0°: Cạnh dài (12cm) nằm ngang theo X+ (hướng phải)")
    print("• Góc 90°: Cạnh dài (12cm) thẳng đứng theo Y+ (hướng xuống)")
    print("• Góc 180°: Cạnh dài (12cm) nằm ngang theo X- (hướng trái)")
    print("• Góc -90°: Cạnh dài (12cm) thẳng đứng theo Y- (hướng lên)")
    print()
    print("NHƯNG: YOLO model có thể định nghĩa khác!")
    print("👉 Hãy chạy 'python rotation_reference_demo.py' với ảnh thật")
    print("   để xem YOLO model của bạn định nghĩa như thế nào!")

def main():
    """Menu chính"""
    print("DEMO ORIENTATION CHO PALLET HÌNH CHỮ NHẬT 12x10cm")
    print("="*60)
    print()
    print("Chương trình này sẽ giải thích:")
    print("1. Cách xác định góc 0°, 90°, 180°, -90° cho pallet hình chữ nhật")
    print("2. Ý nghĩa của trục dài (12cm) và trục ngắn (10cm)")
    print("3. Các góc nghiêng khác (30°, 45°, 125°, ...)")
    print("4. So sánh với YOLO thực tế")
    print()
    
    input("Nhấn Enter để bắt đầu...")
    demo_pallet_orientation()

if __name__ == "__main__":
    main() 