from Camera_Handler import Logger, CameraHandler
from Calibration import Calibration
from Pallet_Detection import PalletDetection
from Module_Division import Module1 # Giả sử có Module2 và Module3
import cv2

class MainApp:
    def __init__(self):
        self.camera = CameraHandler()
        self.processor = PalletDetection()  # Khởi tạo PalletDetection
        self.calibration = Calibration(
            image_points=[[1118, 287], [1118, 709], [635, 709], [635, 287]],
            robot_points=[[-80, 120], [-80, 308], [105, 308], [105, 120]]
        )
        
        # Thiết lập module chia pallet (có thể thay đổi giữa Module1, Module2, Module3)
        self.processor.set_division_module(Module1())  # Mặc định là Module1
        
    def run(self):
        try:
            while True:
                frame = self.camera.capture_frame()
                if frame is None:
                    continue

                # roi_frame = PalletDetection.filter_roi(frame)
                # pallets, processed_frame, edges = self.processor.detect_pallets(roi_frame)
                try:
                    pallets, loads, processed_frame, edges = self.processor.detect_objects(frame)
                except ValueError as e:
                    Logger.log(f"Lỗi trong detect_objects: {str(e)}", debug=True)
                    continue
            
                if pallets:
                    Logger.log("Phát hiện pallet thành công.")
                    for pallet in pallets:
                        try:
                            # Vẽ các điểm chia
                            self.processor.draw_division_points(
                                processed_frame,
                                pallet["box"],
                                lambda pixel: Calibration.pixel_to_robot(pixel, self.calibration.homography_matrix),
                                row=1
                            )
                            
                            # Tính toán tọa độ chia
                            coordinates = self.processor.divide(
                                pallet["box"],
                                1,
                                lambda pixel: Calibration.pixel_to_robot(pixel, self.calibration.homography_matrix)
                            )
                            if coordinates:
                                Logger.log(f"Tọa độ chia: {coordinates}")
                        except Exception as e:
                            Logger.log(f"Lỗi khi xử lý pallet: {e}", debug=True)

                if loads:
                    Logger.log(f"Phát hiện {len(loads)} load")
                    # for load in loads:
                    #     try:
                    #         cv2.drawContours(processed_frame, [load['box']], -1, (0, 0, 255), 2)
                    #         cv2.putText(processed_frame, f"Load {load['center']}", 
                    #                     (load['center'][0] + 10, load['center'][1] + 20), 
                    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    #     except Exception as e:
                    #         Logger.log(f"Lỗi load: {str(e)}", debug=True)

                # Hiển thị tracking
                for obj in self.processor.tracked_objects:
                    Logger.log(f"{obj['type']} tại {obj['center']}", debug=True)
                
                for obj in self.processor.tracked_objects:
                    status = "Mới" if obj['missing_frames'] == 0 else f"Theo dõi ({obj['missing_frames']} frame)"
                    Logger.log(f"{status} - {obj['type']} tại {obj['center']}", debug=True)
                
                
                cv2.imshow("Edges", edges)
                cv2.imshow("Contours", processed_frame)
                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            Logger.log(f"Lỗi chính: {e}")
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            Logger.log("Đã dừng chương trình.")

if __name__ == "__main__":
    app = MainApp()
    app.run()