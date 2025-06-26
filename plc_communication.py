import snap7
import struct
import time
import logging
from typing import Optional, Union, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PLCCommunication:
    """
    Class để communicate với PLC Siemens sử dụng snap7
    """
    
    def __init__(self, ip_address: str = "192.168.0.1", rack: int = 0, slot: int = 1):
        """
        Khởi tạo PLC connection
        
        Args:
            ip_address: IP address của PLC (default: 192.168.0.1)
            rack: Rack number (default: 0, thường là 0 cho S7-1200/1500)
            slot: Slot number (default: 1, thường là 1 cho CPU)
        """
        self.ip_address = ip_address
        self.rack = rack
        self.slot = slot
        self.plc = snap7.client.Client()
        self.connected = False
        
    def connect(self) -> bool:
        """
        Kết nối đến PLC
        
        Returns:
            bool: True nếu kết nối thành công, False nếu thất bại
        """
        try:
            logger.info(f"Đang kết nối đến PLC tại {self.ip_address}, Rack: {self.rack}, Slot: {self.slot}")
            self.plc.connect(self.ip_address, self.rack, self.slot)
            self.connected = True
            logger.info("Kết nối PLC thành công!")
            return True
        except Exception as e:
            logger.error(f"Lỗi kết nối PLC: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """
        Ngắt kết nối PLC
        """
        if self.connected:
            try:
                self.plc.disconnect()
                self.connected = False
                logger.info("Đã ngắt kết nối PLC")
            except Exception as e:
                logger.error(f"Lỗi ngắt kết nối PLC: {e}")
    
    def is_connected(self) -> bool:
        """
        Kiểm tra trạng thái kết nối
        
        Returns:
            bool: True nếu đang kết nối, False nếu không
        """
        try:
            return self.plc.get_connected() if hasattr(self.plc, 'get_connected') else self.connected
        except:
            return False
    
    def read_db_data(self, db_number: int, start_offset: int, size: int) -> Optional[bytearray]:
        """
        Đọc dữ liệu từ Data Block
        
        Args:
            db_number: Số DB (ví dụ: 26 cho DB26)
            start_offset: Vị trí bắt đầu đọc (byte offset)
            size: Số byte cần đọc
            
        Returns:
            bytearray hoặc None nếu lỗi
        """
        if not self.connected:
            logger.error("PLC chưa được kết nối")
            return None
            
        try:
            data = self.plc.db_read(db_number, start_offset, size)
            logger.debug(f"Đọc {size} bytes từ DB{db_number}.{start_offset}")
            return data
        except Exception as e:
            logger.error(f"Lỗi đọc DB{db_number}.{start_offset}: {e}")
            return None
    
    def write_db_data(self, db_number: int, start_offset: int, data: bytearray) -> bool:
        """
        Ghi dữ liệu vào Data Block
        
        Args:
            db_number: Số DB
            start_offset: Vị trí bắt đầu ghi
            data: Dữ liệu cần ghi (bytearray)
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        if not self.connected:
            logger.error("PLC chưa được kết nối")
            return False
            
        try:
            self.plc.db_write(db_number, start_offset, data)
            logger.debug(f"Ghi {len(data)} bytes vào DB{db_number}.{start_offset}")
            return True
        except Exception as e:
            logger.error(f"Lỗi ghi DB{db_number}.{start_offset}: {e}")
            return False
    
    # Các hàm tiện ích để đọc/ghi các kiểu dữ liệu cụ thể
    
    def read_bool(self, db_number: int, byte_offset: int, bit_offset: int) -> Optional[bool]:
        """
        Đọc một bit (BOOL) từ DB
        
        Args:
            db_number: Số DB
            byte_offset: Vị trí byte
            bit_offset: Vị trí bit (0-7)
            
        Returns:
            bool hoặc None nếu lỗi
        """
        data = self.read_db_data(db_number, byte_offset, 1)
        if data is not None:
            return bool(data[0] & (1 << bit_offset))
        return None
    
    def write_bool(self, db_number: int, byte_offset: int, bit_offset: int, value: bool) -> bool:
        """
        Ghi một bit (BOOL) vào DB
        """
        data = self.read_db_data(db_number, byte_offset, 1)
        if data is not None:
            if value:
                data[0] |= (1 << bit_offset)  # Set bit
            else:
                data[0] &= ~(1 << bit_offset)  # Clear bit
            return self.write_db_data(db_number, byte_offset, data)
        return False
    
    def read_int(self, db_number: int, byte_offset: int) -> Optional[int]:
        """
        Đọc INT (16-bit signed integer) từ DB
        """
        data = self.read_db_data(db_number, byte_offset, 2)
        if data is not None:
            return struct.unpack('>h', data)[0]  # Big-endian signed short
        return None
    
    def write_int(self, db_number: int, byte_offset: int, value: int) -> bool:
        """
        Ghi INT (16-bit signed integer) vào DB
        """
        try:
            data = struct.pack('>h', value)  # Big-endian signed short
            return self.write_db_data(db_number, byte_offset, bytearray(data))
        except Exception as e:
            logger.error(f"Lỗi pack INT: {e}")
            return False
    
    def read_dint(self, db_number: int, byte_offset: int) -> Optional[int]:
        """
        Đọc DINT (32-bit signed integer) từ DB
        """
        data = self.read_db_data(db_number, byte_offset, 4)
        if data is not None:
            return struct.unpack('>i', data)[0]  # Big-endian signed int
        return None
    
    def write_dint(self, db_number: int, byte_offset: int, value: int) -> bool:
        """
        Ghi DINT (32-bit signed integer) vào DB
        """
        try:
            data = struct.pack('>i', value)  # Big-endian signed int
            return self.write_db_data(db_number, byte_offset, bytearray(data))
        except Exception as e:
            logger.error(f"Lỗi pack DINT: {e}")
            return False
    
    def read_real(self, db_number: int, byte_offset: int) -> Optional[float]:
        """
        Đọc REAL (32-bit float) từ DB
        """
        data = self.read_db_data(db_number, byte_offset, 4)
        if data is not None:
            return struct.unpack('>f', data)[0]  # Big-endian float
        return None
    
    def write_real(self, db_number: int, byte_offset: int, value: float) -> bool:
        """
        Ghi REAL (32-bit float) vào DB
        """
        try:
            data = struct.pack('>f', value)  # Big-endian float
            return self.write_db_data(db_number, byte_offset, bytearray(data))
        except Exception as e:
            logger.error(f"Lỗi pack REAL: {e}")
            return False
    
    def read_string(self, db_number: int, byte_offset: int, max_length: int = 254) -> Optional[str]:
        """
        Đọc STRING từ DB (Siemens String format)
        """
        data = self.read_db_data(db_number, byte_offset, max_length + 2)
        if data is not None:
            actual_length = data[1]  # Byte 1 chứa độ dài thực tế
            if actual_length <= max_length:
                string_data = data[2:2 + actual_length]
                return string_data.decode('utf-8', errors='ignore')
        return None
    
    def write_string(self, db_number: int, byte_offset: int, value: str, max_length: int = 254) -> bool:
        """
        Ghi STRING vào DB (Siemens String format)
        """
        try:
            encoded = value.encode('utf-8')
            actual_length = min(len(encoded), max_length)
            
            # Tạo string data: [max_length, actual_length, string_bytes, padding]
            data = bytearray(max_length + 2)
            data[0] = max_length  # Byte 0: max length
            data[1] = actual_length  # Byte 1: actual length
            data[2:2 + actual_length] = encoded[:actual_length]  # String data
            
            return self.write_db_data(db_number, byte_offset, data)
        except Exception as e:
            logger.error(f"Lỗi pack STRING: {e}")
            return False

# Class riêng cho DB26
class DB26Communication(PLCCommunication):
    """
    Class chuyên biệt cho communication với DB26
    """
    
    def __init__(self, ip_address: str = "192.168.0.1", rack: int = 0, slot: int = 1):
        super().__init__(ip_address, rack, slot)
        self.db_number = 26
    
    def read_db26_bool(self, byte_offset: int, bit_offset: int) -> Optional[bool]:
        """Đọc BOOL từ DB26"""
        return self.read_bool(self.db_number, byte_offset, bit_offset)
    
    def write_db26_bool(self, byte_offset: int, bit_offset: int, value: bool) -> bool:
        """Ghi BOOL vào DB26"""
        return self.write_bool(self.db_number, byte_offset, bit_offset, value)
    
    def read_db26_int(self, byte_offset: int) -> Optional[int]:
        """Đọc INT từ DB26"""
        return self.read_int(self.db_number, byte_offset)
    
    def write_db26_int(self, byte_offset: int, value: int) -> bool:
        """Ghi INT vào DB26"""
        return self.write_int(self.db_number, byte_offset, value)
    
    def read_db26_dint(self, byte_offset: int) -> Optional[int]:
        """Đọc DINT từ DB26"""
        return self.read_dint(self.db_number, byte_offset)
    
    def write_db26_dint(self, byte_offset: int, value: int) -> bool:
        """Ghi DINT vào DB26"""
        return self.write_dint(self.db_number, byte_offset, value)
    
    def read_db26_real(self, byte_offset: int) -> Optional[float]:
        """Đọc REAL từ DB26"""
        return self.read_real(self.db_number, byte_offset)
    
    def write_db26_real(self, byte_offset: int, value: float) -> bool:
        """Ghi REAL vào DB26"""
        return self.write_real(self.db_number, byte_offset, value)

def test_connection():
    """
    Hàm test kết nối PLC với các rack/slot khác nhau
    """
    ip_address = "192.168.0.1"
    
    # Các rack/slot phổ biến để thử
    rack_slot_combinations = [
        (0, 1),  # S7-1200/1500 CPU
        (0, 2),  # S7-1200/1500 CPU slot 2
        (0, 0),  # S7-300/400
        (1, 0),  # S7-300/400 rack 1
    ]
    
    for rack, slot in rack_slot_combinations:
        print(f"\nThử kết nối với Rack: {rack}, Slot: {slot}")
        plc = PLCCommunication(ip_address, rack, slot)
        
        if plc.connect():
            print(f"✓ Kết nối thành công với Rack: {rack}, Slot: {slot}")
            plc.disconnect()
            return rack, slot
        else:
            print(f"✗ Không thể kết nối với Rack: {rack}, Slot: {slot}")
    
    print("\nKhông thể kết nối với bất kỳ rack/slot nào!")
    return None, None

if __name__ == "__main__":
    # Test kết nối và tìm rack/slot đúng
    print("=== Test kết nối PLC ===")
    rack, slot = test_connection()
    
    if rack is not None and slot is not None:
        print(f"\n=== Sử dụng DB26 với Rack: {rack}, Slot: {slot} ===")
        
        # Tạo instance DB26
        db26 = DB26Communication("192.168.0.1", rack, slot)
        
        if db26.connect():
            try:
                # Ví dụ đọc/ghi dữ liệu DB26
                print("\n--- Test đọc/ghi DB26 ---")
                
                # Test BOOL
                print("Test BOOL (DB26.0.0):")
                bool_value = db26.read_db26_bool(0, 0)
                print(f"Giá trị hiện tại: {bool_value}")
                
                # Test INT
                print("\nTest INT (DB26.2):")
                int_value = db26.read_db26_int(2)
                print(f"Giá trị hiện tại: {int_value}")
                
                # Test DINT
                print("\nTest DINT (DB26.4):")
                dint_value = db26.read_db26_dint(4)
                print(f"Giá trị hiện tại: {dint_value}")
                
                # Test REAL
                print("\nTest REAL (DB26.8):")
                real_value = db26.read_db26_real(8)
                print(f"Giá trị hiện tại: {real_value}")
                
                # Ghi test values (uncomment để test ghi)
                # print("\n--- Test ghi dữ liệu ---")
                # db26.write_db26_bool(0, 0, True)
                # db26.write_db26_int(2, 1234)
                # db26.write_db26_dint(4, 56789)
                # db26.write_db26_real(8, 3.14159)
                
            except Exception as e:
                print(f"Lỗi trong quá trình test: {e}")
            finally:
                db26.disconnect()
        else:
            print("Không thể kết nối đến DB26")
    else:
        print("Vui lòng kiểm tra:")
        print("1. PLCSIM đã chạy chưa?")
        print("2. IP address có đúng không? (hiện tại: 192.168.0.1)")
        print("3. Firewall có block kết nối không?") 