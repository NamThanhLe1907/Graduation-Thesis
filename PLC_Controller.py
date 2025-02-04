import snap7
from snap7.util import *
import time
from typing import Tuple, Optional

class PLCController:
    def __init__(self, ip: str, rack: int, slot: int, db_read: int, db_write: int):
        self.ip = ip
        self.rack = rack
        self.slot = slot
        self.db_read = db_read
        self.db_write = db_write
        self.client = snap7.client.Client()
        
        # Kết nối PLC
        try:
            self.client.connect(self.ip, self.rack, self.slot)
            print(f"Connected to PLC at {self.ip}")
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            raise

    def __del__(self):
        if self.client.get_connected():
            self.client.disconnect()
            print("PLC disconnected")

    def read_hang_bao_from_db(self) -> Tuple[Optional[int], Optional[int], Optional[bool]]:
        """Đọc giá trị hàng, bao và tín hiệu done từ DB đọc"""
        try:
            # Đọc 5 bytes từ DB (giả sử cấu trúc: 2 bytes hàng, 2 bytes bao, 1 byte done)
            data = self.client.db_read(self.db_read, 0, 5)
            
            hang = get_int(data, 0)    # Offset 0, kiểu INT
            bao = get_int(data, 2)     # Offset 2, kiểu INT
            done = get_bool(data, 4, 0) # Offset 4, bit 0
            
            return hang, bao, done
        except Exception as e:
            print(f"Read error: {str(e)}")
            return None, None, None

    def write_to_db38(self, x: float, y: float) -> bool:
        """Ghi tọa độ X,Y vào DB ghi"""
        try:
            data = bytearray(8)  # 2 biến REAL (4 bytes mỗi cái)
            
            # Convert giá trị float sang bytes
            set_real(data, 0, x)  # Offset 0
            set_real(data, 4, y)  # Offset 4
            
            self.client.db_write(self.db_write, 0, data)
            return True
        except Exception as e:
            print(f"Write coordinates error: {str(e)}")
            return False

    def write_done_to_db(self, done_value: bool) -> bool:
        """Ghi tín hiệu done vào DB"""
        try:
            data = bytearray(1)  # 1 byte
            set_bool(data, 0, 0, done_value)  # Byte 0, bit 0
            self.client.db_write(self.db_write, 8, data)  # Giả sử offset 8
            return True
        except Exception as e:
            print(f"Write done error: {str(e)}")
            return False

    def wait_for_done_signal(self, timeout: float = 30.0) -> bool:
        """Chờ tín hiệu done từ PLC"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            _, _, done = self.read_hang_bao_from_db()
            if done:
                return True
            time.sleep(0.1)  # Tránh chiếm CPU
        return False

# # Cách sử dụng
# if __name__ == "__main__":
#     plc = PLCController("192.168.0.1", 0, 1, 63, 38)
    
#     # Đọc dữ liệu
#     hang, bao, done = plc.read_hang_bao_from_db()
#     print(f"Hàng: {hang}, Bao: {bao}, Done: {done}")
    
#     # Ghi tọa độ
#     if plc.write_to_db38(123.45, 678.90):
#         print("Ghi tọa độ thành công!")
    
#     # Chờ tín hiệu done
#     if plc.wait_for_done_signal():
#         print("Nhận được tín hiệu done!")
#     else:
#         print("Timeout!")