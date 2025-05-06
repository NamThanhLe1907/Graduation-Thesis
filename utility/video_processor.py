from utility import (CameraInterface,
                    QueueManager
                    )



class FrameCamera:
    def __init__(self,camera):
        self.camera = camera 
        self.frame_queue = QueueManager(maxsize = 30)
        self.running = False 
        self.frame_counter = 0
        
    