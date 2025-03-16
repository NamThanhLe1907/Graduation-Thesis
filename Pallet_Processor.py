from .Camera_Handler import Logger
from Calibration import Calibration
from Pallet_Detection import PalletDetection
from PLC_Controller import PLCController
from Module_Division import DivisionModule



class PalletProcessor:
    def __init__(self, plc_controller):
        self.plc = PLCController

    def process_frame(self, frame, hang, bao):
        pallets, processed_frame, edges = PalletDetection.detect_pallets(frame)

        if not pallets:
            Logger.log("Không phát hiện pallet nào.")
            return processed_frame, edges, False

        for pallet in pallets:
            try:
                DivisionModule.draw_division_points(processed_frame, pallet, hang, Calibration.pixel_to_robot)
            except Exception as e:
                Logger.log(f"Lỗi khi vẽ đường phân chia: {e}", debug=True)

            coordinates, _ = PalletDetection.divide_pallet_by_row(pallet, hang, Calibration.pixel_to_robot)

            if bao < 1 or bao > len(coordinates):
                Logger.log(f"Lỗi: Bao ({bao}) không hợp lệ.")
                return processed_frame, edges, False

            x, y = coordinates[bao - 1]

            if self.plc.write_data_38(x, y):
                Logger.log(f"Ghi thành công tọa độ: X = {x:.3f}, Y = {y:.3f}")
                self.plc.write_done_to_db(0)
                return processed_frame, edges, True

        return processed_frame, edges, False