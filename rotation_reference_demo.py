"""
Demo chi tiết về hệ quy chiếu góc xoay trong Computer Vision.
Giải thích cách YOLO OBB xác định góc và cách robot cần xoay.
"""
import cv2
import numpy as np
import math
import os
from detection import YOLOTensorRT, ModuleDivision
from detection.utils.rotation_analyzer import RotationAnalyzer

# Đường dẫn model
ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "utils", "detection", "best.engine")

def draw_coordinate_system(image, center, scale=100):
    """
    Vẽ hệ tọa độ tại một điểm để hiểu hướng của các trục.
    
    Args:
        image: Ảnh để vẽ lên
        center: Điểm trung tâm (x, y)
        scale: Độ dài của trục
    """
    cx, cy = int(center[0]), int(center[1])
    
    # Vẽ trục X (màu đỏ) - hướng phải
    cv2.arrowedLine(image, (cx, cy), (cx + scale, cy), (0, 0, 255), 3, tipLength=0.3)
    cv2.putText(image, "X+", (cx + scale + 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Vẽ trục Y (màu xanh lá) - hướng xuống
    cv2.arrowedLine(image, (cx, cy), (cx, cy + scale), (0, 255, 0), 3, tipLength=0.3)
    cv2.putText(image, "Y+", (cx + 5, cy + scale + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Vẽ điểm gốc
    cv2.circle(image, (cx, cy), 5, (255, 255, 255), -1)
    cv2.putText(image, "O", (cx - 15, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_angle_arc(image, center, angle_deg, radius=60, color=(255, 255, 0)):
    """
    Vẽ cung tròn biểu thị góc từ trục X dương.
    
    Args:
        image: Ảnh để vẽ lên
        center: Điểm trung tâm
        angle_deg: Góc (độ)
        radius: Bán kính cung tròn
        color: Màu của cung
    """
    cx, cy = int(center[0]), int(center[1])
    
    # Vẽ cung tròn từ 0° đến angle_deg
    start_angle = 0
    end_angle = int(angle_deg)
    
    if angle_deg >= 0:
        # Góc dương: ngược chiều kim đồng hồ
        cv2.ellipse(image, (cx, cy), (radius, radius), 0, start_angle, end_angle, color, 2)
    else:
        # Góc âm: cùng chiều kim đồng hồ  
        cv2.ellipse(image, (cx, cy), (radius, radius), 0, end_angle, start_angle, color, 2)
    
    # Vẽ text góc
    text_x = cx + radius + 10
    text_y = cy
    cv2.putText(image, f"{angle_deg:.1f}°", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_object_with_detailed_info(image, obj_info, show_details=True):
    """
    Vẽ object với thông tin chi tiết về góc và hướng.
    
    Args:
        image: Ảnh để vẽ lên
        obj_info: Thông tin object từ rotation analyzer
        show_details: Có hiển thị thông tin chi tiết không
    """
    cx, cy = obj_info['center']
    angle_deg = obj_info['angle_normalized']
    class_name = obj_info['class_name']
    
    # Màu theo class
    if 'load' in class_name.lower():
        if 'load_2' in class_name.lower():
            color = (255, 0, 255)  # Tím cho load_2
        else:
            color = (0, 255, 0)    # Xanh lá cho load
    else:
        color = (255, 0, 0)        # Đỏ cho pallet
    
    # Vẽ điểm trung tâm
    cv2.circle(image, (int(cx), int(cy)), 8, color, -1)
    cv2.circle(image, (int(cx), int(cy)), 12, (255, 255, 255), 2)
    
    if show_details:
        # Vẽ hệ tọa độ tại object
        draw_coordinate_system(image, (cx, cy), scale=40)
        
        # Vẽ cung góc
        draw_angle_arc(image, (cx, cy), angle_deg, radius=50, color=color)
    
    # Vẽ vector hướng của object
    length = 80
    end_x = cx + length * math.cos(math.radians(angle_deg))
    end_y = cy + length * math.sin(math.radians(angle_deg))
    cv2.arrowedLine(image, (int(cx), int(cy)), (int(end_x), int(end_y)), color, 4, tipLength=0.2)
    
    # Text thông tin
    info_text = f"{class_name}: {angle_deg:.1f}°"
    
    # Xác định quadrant và hướng
    if -5 <= angle_deg <= 5:
        direction = "→ (East)"
    elif 85 <= angle_deg <= 95:
        direction = "↓ (South)"
    elif 175 <= angle_deg <= 180 or -180 <= angle_deg <= -175:
        direction = "← (West)"
    elif -95 <= angle_deg <= -85:
        direction = "↑ (North)"
    elif 0 < angle_deg < 90:
        direction = "↘ (SE)"
    elif 90 < angle_deg < 180:
        direction = "↙ (SW)"
    elif -90 < angle_deg < 0:
        direction = "↗ (NE)"
    elif -180 < angle_deg < -90:
        direction = "↖ (NW)"
    else:
        direction = "?"
    
    # Vẽ text với background
    text_lines = [
        info_text,
        f"Hướng: {direction}",
        f"Quadrant: {get_quadrant_name(angle_deg)}"
    ]
    
    y_offset = int(cy) - 60
    for i, line in enumerate(text_lines):
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Background
        cv2.rectangle(image, 
                     (int(cx) - 5, y_offset + i*20 - text_size[1] - 2),
                     (int(cx) + text_size[0] + 5, y_offset + i*20 + 2),
                     (0, 0, 0), -1)
        
        # Text
        cv2.putText(image, line, (int(cx), y_offset + i*20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def get_quadrant_name(angle_deg):
    """Trả về tên quadrant"""
    if 0 <= angle_deg < 90:
        return "IV"
    elif 90 <= angle_deg < 180:
        return "III"
    elif -180 <= angle_deg < -90:
        return "II"
    elif -90 <= angle_deg < 0:
        return "I"
    else:
        return "Boundary"

def create_reference_explanation():
    """
    Tạo ảnh giải thích hệ quy chiếu.
    """
    # Tạo ảnh trắng
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(img, "HE QUY CHIEU GOC XOAY TRONG COMPUTER VISION", 
               (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Vẽ hệ tọa độ chính
    center = (400, 300)
    draw_coordinate_system(img, center, scale=120)
    
    # Giải thích các hướng
    directions = [
        (0, "0° (East, →)", (0, 255, 0)),
        (90, "90° (South, ↓)", (0, 255, 255)),
        (180, "180° (West, ←)", (255, 0, 0)),
        (-90, "-90° (North, ↑)", (255, 0, 255))
    ]
    
    for angle, label, color in directions:
        # Vẽ vector
        length = 100
        end_x = center[0] + length * math.cos(math.radians(angle))
        end_y = center[1] + length * math.sin(math.radians(angle))
        cv2.arrowedLine(img, center, (int(end_x), int(end_y)), color, 3, tipLength=0.2)
        
        # Vẽ cung góc
        if angle != 0:
            draw_angle_arc(img, center, angle, radius=80, color=color)
        
        # Label
        label_x = int(end_x + 20 * math.cos(math.radians(angle)))
        label_y = int(end_y + 20 * math.sin(math.radians(angle)))
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Thêm chú thích
    notes = [
        "LUU Y:",
        "- Goc duong (+): Nguoc chieu kim dong ho (CCW)",
        "- Goc am (-): Cung chieu kim dong ho (CW)",
        "- Goc do tu truc X duong (huong phai)",
        "- He toa do: X+ sang phai, Y+ xuong duoi"
    ]
    
    for i, note in enumerate(notes):
        cv2.putText(img, note, (50, 500 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return img

def demo_rotation_reference():
    """
    Demo chi tiết về hệ quy chiếu góc xoay.
    """
    print("=== DEMO HỆ QUY CHIẾU GÓC XOAY ===")
    
    # Hiển thị ảnh giải thích
    reference_img = create_reference_explanation()
    cv2.imshow("He Quy Chieu Goc Xoay", reference_img)
    
    print("Hiển thị hệ quy chiếu. Nhấn phím bất kỳ để tiếp tục...")
    cv2.waitKey(0)
    
    # Test với ảnh thực
    test_image_path = "images_pallets"
    if os.path.exists(test_image_path):
        image_files = [f for f in os.listdir(test_image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            print(f"\nChọn ảnh để phân tích góc:")
            for i, img_file in enumerate(image_files[:10], 1):  # Hiển thị 10 ảnh đầu
                print(f"  {i}. {img_file}")
            
            choice = input(f"\nChọn ảnh (1-{min(10, len(image_files))}): ")
            try:
                choice_num = int(choice) - 1
                if 0 <= choice_num < len(image_files):
                    image_path = os.path.join(test_image_path, image_files[choice_num])
                else:
                    image_path = os.path.join(test_image_path, image_files[0])
            except:
                image_path = os.path.join(test_image_path, image_files[0])
            
            # Xử lý ảnh
            process_image_with_reference(image_path)
    
    cv2.destroyAllWindows()

def process_image_with_reference(image_path):
    """
    Xử lý ảnh với visualization chi tiết về góc.
    """
    print(f"\nĐang xử lý: {image_path}")
    
    # Khởi tạo models
    yolo_model = YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.25)
    analyzer = RotationAnalyzer(debug=True)
    
    # Đọc ảnh
    frame = cv2.imread(image_path)
    if frame is None:
        print("Không thể đọc ảnh!")
        return
    
    print(f"Kích thước ảnh: {frame.shape[1]} x {frame.shape[0]}")
    
    # YOLO detection
    print("\nThực hiện YOLO detection...")
    detections = yolo_model.detect(frame)
    
    print(f"Phát hiện {len(detections.get('bounding_boxes', []))} objects")
    print(f"Class IDs: {detections.get('classes', [])}")
    print(f"Class Names: {detections.get('class_names', [])}")
    
    # Phân tích góc
    yolo_angles = analyzer.analyze_yolo_angles(detections)
    
    # Tạo 3 ảnh visualization
    
    # 1. Ảnh gốc với YOLO detection
    annotated_frame = detections['annotated_frame'].copy()
    
    # 2. Ảnh với thông tin góc đơn giản
    simple_frame = frame.copy()
    for obj in yolo_angles:
        draw_object_with_detailed_info(simple_frame, obj, show_details=False)
    
    # 3. Ảnh với thông tin góc chi tiết
    detailed_frame = frame.copy()
    for obj in yolo_angles:
        draw_object_with_detailed_info(detailed_frame, obj, show_details=True)
    
    # Hiển thị kết quả
    print(f"\n=== THÔNG TIN CHI TIẾT CÁC OBJECTS ===")
    for i, obj in enumerate(yolo_angles):
        print(f"\nObject {i+1}:")
        print(f"  Class: {obj['class_name']}")
        print(f"  Center: ({obj['center'][0]:.1f}, {obj['center'][1]:.1f})")
        print(f"  Góc gốc: {obj['angle_deg']:.1f}°")
        print(f"  Góc chuẩn hóa: {obj['angle_normalized']:.1f}°")
        print(f"  Hướng: {obj['angle_info']['direction']}")
        print(f"  Quadrant: {get_quadrant_name(obj['angle_normalized'])}")
        
        # Giải thích ý nghĩa góc
        angle = obj['angle_normalized']
        if -5 <= angle <= 5:
            print(f"  → Object đang nằm NGANG (horizontal)")
        elif 85 <= angle <= 95:
            print(f"  → Object đang THẲNG ĐỨNG hướng xuống")
        elif 175 <= angle <= 180 or -180 <= angle <= -175:
            print(f"  → Object đang nằm NGANG ngược lại")
        elif -95 <= angle <= -85:
            print(f"  → Object đang THẲNG ĐỨNG hướng lên")
        else:
            print(f"  → Object đang NGHIÊNG {angle:.1f}°")
    
    # Hiển thị các ảnh
    cv2.imshow("1. YOLO Detection Goc", annotated_frame)
    cv2.imshow("2. Goc Xoay Don Gian", simple_frame)
    cv2.imshow("3. Goc Xoay Chi Tiet", detailed_frame)
    
    print(f"\nCác ảnh đã được hiển thị:")
    print(f"1. YOLO Detection Gốc - Kết quả phát hiện từ YOLO")
    print(f"2. Góc Xoay Đơn Giản - Arrows và thông tin cơ bản")
    print(f"3. Góc Xoay Chi Tiết - Hệ tọa độ, cung góc, và thông tin đầy đủ")
    
    print(f"\nNhấn phím bất kỳ để tiếp tục...")
    cv2.waitKey(0)
    
    # Lưu kết quả
    save_choice = input("\nBạn có muốn lưu kết quả? (y/n): ").lower()
    if save_choice in ['y', 'yes']:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        cv2.imwrite(f"reference_yolo_{base_name}.jpg", annotated_frame)
        cv2.imwrite(f"reference_simple_{base_name}.jpg", simple_frame)
        cv2.imwrite(f"reference_detailed_{base_name}.jpg", detailed_frame)
        cv2.imwrite(f"reference_explanation.jpg", create_reference_explanation())
        
        print(f"Đã lưu 4 ảnh:")
        print(f"  - reference_yolo_{base_name}.jpg")
        print(f"  - reference_simple_{base_name}.jpg") 
        print(f"  - reference_detailed_{base_name}.jpg")
        print(f"  - reference_explanation.jpg")

def main():
    """Menu chính"""
    print("DEMO HỆ QUY CHIẾU GÓC XOAY TRONG COMPUTER VISION")
    print("="*60)
    print()
    print("Chương trình này sẽ giải thích:")
    print("1. Hệ quy chiếu góc trong Computer Vision")
    print("2. Cách YOLO OBB xác định góc xoay")
    print("3. Ý nghĩa của các góc dương/âm") 
    print("4. Cách object hướng theo các góc khác nhau")
    print()
    print("Giải thích hệ quy chiếu:")
    print("- Trục X+: Hướng sang PHẢI")
    print("- Trục Y+: Hướng XUỐNG DƯỚI")
    print("- Góc 0°: Hướng SANG PHẢI (East)")
    print("- Góc +90°: Hướng XUỐNG DƯỚI (South)")
    print("- Góc ±180°: Hướng SANG TRÁI (West)")
    print("- Góc -90°: Hướng LÊN TRÊN (North)")
    print("- Góc dương (+): Ngược chiều kim đồng hồ (CCW)")
    print("- Góc âm (-): Cùng chiều kim đồng hồ (CW)")
    print()
    
    input("Nhấn Enter để bắt đầu demo...")
    demo_rotation_reference()

if __name__ == "__main__":
    main() 