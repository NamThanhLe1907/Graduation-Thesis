import unittest
import cv2
import numpy as np
import sys
import tkinter as tk
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.visualization import AppGUI

class TestAppGUI(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.app_gui = AppGUI(self.root)
        self.test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def tearDown(self):
        self.root.destroy()

    def test_update_method(self):
        try:
            # Thử cập nhật frame trống
            self.app_gui.update(self.test_frame)
        except Exception as e:
            self.fail(f"Lỗi khi gọi phương thức update: {e}")

    def test_log_message_method(self):
        try:
            # Thử ghi log các loại thông điệp
            self.app_gui.log_message("Thông điệp thông thường")
            self.app_gui.log_message("Cảnh báo", "WARNING")
            self.app_gui.log_message("Lỗi nghiêm trọng", "ERROR")
        except Exception as e:
            self.fail(f"Lỗi khi ghi log: {e}")

if __name__ == "__main__":
    unittest.main()