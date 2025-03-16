import cv2
import numpy as np
from ultralytics import YOLO
from Camera_Handler import CameraHandler, Logger
from Module_Division import Module1

def main():
    # Load mô hình YOLO đã được train cho OBB (Oriented Bounding Boxes)
    model = YOLO('best.pt')
    # Khởi tạo Camera với độ phân giải 1280x1024
    camera = CameraHandler()
    # Module chia pallet không được dùng trong phần demo này
    division_module = Module1()
    row = 1  # Số hàng (không ảnh hưởng tới OBB)

    while True:
        frame = camera.capture_frame()
        if frame is None:
            continue

        # Dự đoán bằng YOLO với ngưỡng confidence 0.90
        results = model(frame, conf=0.90)
        # Sử dụng bản sao của frame gốc để vẽ rotated bounding boxes
        annotated_frame = frame.copy()

        # Kiểm tra và xử lý OBB detections (dựa vào results[0].obb)
        if results and results[0].obb is not None and len(results[0].obb.data) > 0:
            # Giả sử mỗi dự đoán có dạng: [center_x, center_y, w, h, angle, conf, cls]
            obb_preds = results[0].obb.data.cpu().numpy()
            for pred in obb_preds:
                try:
                    center_x, center_y, w, h, angle, conf, cls = map(float, pred)
                    center = (int(center_x), int(center_y))
                    
                    # Xây dựng rotated rectangle dựa trên OBB
                    rect = (center, (w, h), angle)
                    box_points = cv2.boxPoints(rect)
                    box_points = np.int32(box_points)
                    
                    # Vẽ rotated bounding box lên annotated_frame
                    cv2.polylines(annotated_frame, [box_points], isClosed=True, color=(0, 255, 0), thickness=2)
                    # Vẽ điểm trung tâm
                    cv2.circle(annotated_frame, center, 5, (255, 0, 0), -1)
                    # In thông tin center và góc xoay lên ảnh
                    text = f"Center: {center}, Angle: {angle:.1f} deg"
                    cv2.putText(annotated_frame, text, (center[0]-50, center[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Log thông tin detection
                    Logger.log(f"Detection - Center: {center}, Angle: {angle:.1f} deg, Class: {int(cls)}", debug=True)
                except Exception as e:
                    Logger.log(f"Error processing OBB detection: {e}", debug=True)
        else:
            Logger.log("No OBB detections", debug=True)

        cv2.imshow("YOLO OBB Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
