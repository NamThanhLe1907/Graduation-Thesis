"""
Các tiện ích hỗ trợ testing
"""
import multiprocessing as mp

# Đảm bảo MockCamera ở top-level module để có thể pickle
class MockCamera:
    """Camera giả lập – trả chuỗi 'frame_1', 'frame_2', …"""
    def __init__(self):
        self._n = 0

    def get_frame(self):
        self._n += 1
        return f"frame_{self._n}"


# Đảm bảo factory function ở top-level để có thể pickle
def create_mock_camera():
    """Factory function để tạo MockCamera - dùng cho multiprocessing"""
    return MockCamera()


# Hàm worker PHẢI ở top-level cho Windows multiprocessing
def pipe_worker(conn):
    """Worker gửi message qua pipe"""
    conn.send("Hello from worker")
    conn.close()


# Thử phương pháp khác - sử dụng pipe thay vì Queue
def test_mp_pipe():
    """Test xem multiprocessing cơ bản có hoạt động không"""
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=pipe_worker, args=(child_conn,))
    p.start()
    print(f"Received: {parent_conn.recv()}")
    p.join()
    return True 