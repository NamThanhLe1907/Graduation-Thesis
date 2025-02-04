import cv2
from time import sleep


DEBUG = True

class Logger:
    @staticmethod
    def log(message, debug=False):
        if DEBUG or not debug:
            print(f"[LOG] {message}")
        
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