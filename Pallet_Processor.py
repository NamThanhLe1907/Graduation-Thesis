from Camera_Handler import Logger
from Pallet_Detection import (
    find_pallet_in_list,
    apply_roi,
    detect_pallets,
    draw_division_points,
    divide_pallet_by_row,
    pixel_to_robot
)
from PLC_Controller import (
    write_done_to_db, 
    write_done_to_db,
    write_data_38
)


class PalletProcessor:
    def __init__(self, plc_controller):
        self.plc = plc_controller

    def process_frame(self, frame, hang, bao):
        roi_frame = apply_roi(frame)
        pallets, processed_frame, edges = detect_pallets(roi_frame)

        if not pallets:
            Logger.log("Không phát hiện pallet nào.")
            return processed_frame, edges, False

        for pallet in pallets:
            try:
                draw_division_points(processed_frame, pallet, hang, pixel_to_robot)
            except Exception as e:
                Logger.log(f"Lỗi khi vẽ đường phân chia: {e}", debug=True)

            coordinates, _ = divide_pallet_by_row(pallet, hang, pixel_to_robot)

            if bao < 1 or bao > len(coordinates):
                Logger.log(f"Lỗi: Bao ({bao}) không hợp lệ.")
                return processed_frame, edges, False

            x, y = coordinates[bao - 1]

            if self.plc.write_data_38(x, y):
                Logger.log(f"Ghi thành công tọa độ: X = {x:.3f}, Y = {y:.3f}")
                self.plc.write_done_to_db(0)
                return processed_frame, edges, True

        return processed_frame, edges, False