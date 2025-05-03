from queue import Queue, Full, Empty 
from utility import (YOLOInference,
                     DepthEstimator)


class QueueManger:
    """
    - put(item): ghi đè image cũ
    - get(timeout): lấy image, trả về None nếu timeout
    - task_done(): biến báo đã xử lý xong image
    - clear(): xóa hàng đợi
    """ 

    def __init__(self):
        self._q = Queue(maxsize= 1)
        
    #Producer
    def put(self,image):
        try:
            self._q.put_nowait(image)
        except Full:
            try: 
                self._q.get_nowait()
            except Empty:
                pass
            self._q.put_nowait(image)
        return True
    
    #Consumer
    
    def get(self, timeout = None):
        try:
            return self._q.get(timeout=timeout)
        except Empty:
            return None
        
    def task_done(self):
        self._q.task_done()
    
    def clear(self):
        while  not self._q.empty():
            try:
                self._q.get_nowait()
            except Empty:
                break
            