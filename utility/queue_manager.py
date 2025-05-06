from multiprocessing import Queue
from queue import Full, Empty 
from typing import Any, Optional

class QueueManager:
    """Quản lý một FIFO queue dùng giữa các process.

    Parameters
    ----------
    maxsize : int, default 1
        Số phần tử tối đa. Khi đầy, phần tử cũ nhất sẽ bị loại bỏ để nhường
        chỗ cho phần tử mới (overwrite‑on‑full). Điều này giúp producer không
        bị block và luôn xử lý frame mới nhất.
    """

    def __init__(self, maxsize: int = 1):
        self._q: Queue[Any] = Queue(maxsize = maxsize)
        
    #Producer
    def put(self,item: Any) -> bool:
        """Đưa *item*  vào Queue và thay cũ"""
        try:
            self._q.put_nowait(item)
        except Exception:
            try:
                self._q.get_nowait()
            except Empty:
                pass 
            self._q.put_nowait(item)
        return True
    
    #Consumer
    
    def get(self, timeout: Optional[float] = None) -> Any:
        """Lấy và xóa một item khỏi queue.
        Trả về ``None`` nếu hết thời gian chờ mà không có dữ liệu.
        """
        try:
            return self._q.get(timeout=timeout)
        except Empty:
            return None
        
    def clear(self) -> None:
        """
        Xóa sạch các item còn lại trong queue.
        """
        while True:     
            try:
                self._q.get_nowait()
            except Empty:
                break
            
    def empty(self) -> bool:
        try:
            return self._q.empty()
        except NotImplementedError:
            return False
        
    def full(self) -> bool:
        try:
            return self._q.full()
        except  NotImplementedError:
            return False
        
    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except NotImplementedError:
            return 0