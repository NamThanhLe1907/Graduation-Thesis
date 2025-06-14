"""
Test cơ bản cho multiprocessing để cô lập vấn đề
"""
import unittest
import multiprocessing as mp
from test_utils import test_mp_pipe, MockCamera, create_mock_camera


def simple_worker(queue):
    """Worker đơn giản gửi một số giá trị vào queue"""
    queue.put("test1")
    queue.put("test2")
    queue.put("test3")


def camera_worker(queue):
    """Worker tạo MockCamera và gửi frame vào queue"""
    camera = create_mock_camera()
    for _ in range(5):
        frame = camera.get_frame()
        queue.put(frame)


class TestBasicMP(unittest.TestCase):
    def test_pipe(self):
        """Test xem pipe cơ bản có hoạt động không"""
        self.assertTrue(test_mp_pipe())

    def test_queue(self):
        """Test xem queue cơ bản có hoạt động không"""
        q = mp.Queue()
        p = mp.Process(target=simple_worker, args=(q,))
        p.start()
        
        items = []
        deadline = mp.Value('d', 0.0)
        with deadline.get_lock():
            deadline.value = mp.current_process().pid
        
        for _ in range(3):
            items.append(q.get(timeout=1))
        
        p.join()
        
        self.assertEqual(items, ["test1", "test2", "test3"])

    def test_camera_queue(self):
        """Test xem camera có serialize được qua multiprocessing không"""
        q = mp.Queue()
        p = mp.Process(target=camera_worker, args=(q,))
        p.start()
        
        items = []
        for _ in range(5):
            items.append(q.get(timeout=1))
        
        p.join()
        
        self.assertEqual(len(items), 5)
        for i, item in enumerate(items, 1):
            self.assertEqual(item, f"frame_{i}")


if __name__ == "__main__":
    unittest.main() 