import cv2

class CameraInterface:
    def __init__(self, camera_index=0, resolution=(1280, 1024), fps=30):
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.capture = None

    def initialize(self):
        self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            # Thử index camera khác nếu mặc định không hoạt động
            self.capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Thử index 1
            if not self.capture.isOpened():
                raise RuntimeError(f"Không thể mở camera tại index {self.camera_index} hoặc 1")
                
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Kiểm tra thông số camera thực tế
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}FPS")

    def get_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            raise Exception("Failed to capture frame")
        return frame

    def release(self):
        if self.capture is not None:
            self.capture.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    camera = CameraInterface()
    camera.initialize()
    while True:
        frame = camera.get_frame()
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()