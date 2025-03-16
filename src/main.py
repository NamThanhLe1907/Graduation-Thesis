from Camera_Handler import Logger, CameraHandler
from Pallet_Processor import PalletProcessor
from PLC_Controller import PLCController
import cv2


class MainApp:
    def __init__(self):
        self.camera = CameraHandler()
        #self.plc = PLCController("192.168.0.1", 0, 1, 63, 38)
        self.processor = PalletProcessor(self.plc)

    def run(self):
        try:
            while True:
                bao, hang, _ = self.plc.read_data()
                if bao is None or hang is None:
                    Logger.log("Lỗi: Không thể đọc dữ liệu từ PLC.")
                    break

                frame = self.camera.capture_frame()
                if frame is None:
                    break

                processed_frame, edges, success = self.processor.process_frame(frame, hang, bao)

                if success:
                    Logger.log("Xử lý thành công bao hiện tại.")
                    bao += 1
                    if bao > 3:
                        bao = 1
                        hang += 1
                    self.plc.write_done_to_db(0)
                else:
                    Logger.log("Chờ tín hiệu 'Done' từ PLC để tiếp tục...")
                    if self.plc.wait_for_done_signal():
                        bao += 1
                        if bao > 3:
                            bao = 1
                            hang += 1

                cv2.imshow("Edges", edges)
                cv2.imshow("Contours", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            Logger.log(f"Lỗi: {e}")
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            Logger.log("Đã dừng chương trình.")


if __name__ == "__main__":
    app = MainApp()
    app.run()