import cv2

class CameraInterface:
    def __init__(self, camera_index=0, resolution=(1280, 1024), fps=30):
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.capture = None

    def initialize(self):
        self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

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