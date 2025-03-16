import cv2
from time import sleep
import time

DEBUG = True

class Logger:
    last_messages = {}  # Dictionary để lưu message và thời gian xuất hiện gần nhất
    log_interval = 2  # Chỉ log lại sau 2 giây nếu cùng nội dung xuất hiện

    @staticmethod
    def log(message, debug=False):
        global DEBUG
        if DEBUG or not debug:
            current_time = time.time()

            # Kiểm tra nếu message này đã log gần đây
            if message in Logger.last_messages:
                last_time = Logger.last_messages[message]
                if current_time - last_time < Logger.log_interval:
                    return  # Bỏ qua log nếu chưa đủ thời gian log_interval

            # Nếu không bị trùng, ghi log và cập nhật thời gian log gần nhất
            print(f"[LOG] {message}")
            Logger.last_messages[message] = current_time

            # Giữ lại tối đa 50 log gần nhất để tránh tràn bộ nhớ
            if len(Logger.last_messages) > 50:
                oldest_key = min(Logger.last_messages, key=Logger.last_messages.get)
                del Logger.last_messages[oldest_key]


        
class CameraHandler:
    def __init__(self, width=1920, height=1080):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            Logger.log("Can not open the camera. Check your connection or driver.")
            raise Exception("Camera is not working.")

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            Logger.log("The device can not read frame from camera.")
            return None
        Logger.log("The device read frames from camera successfully.")
        return frame

    def release(self):
        self.cap.release()