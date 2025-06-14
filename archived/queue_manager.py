"""
QueueManager – wrapper cực mỏng cho multiprocessing.Queue
với hành vi *overwrite‑on‑full* (bỏ frame cũ, giữ frame mới).
Chạy an toàn cả trên Windows (spawn) lẫn Linux (fork).
"""
from __future__ import annotations

import multiprocessing as mp
from queue import Empty, Full
from typing import Any, Optional


class QueueManager:
    def __init__(self, maxsize: int = 1) -> None:
        # Lấy context phù hợp (spawn / fork / forkserver)
        ctx = mp.get_context()
        self._q: mp.Queue[Any] = ctx.Queue(maxsize=maxsize)

    # ---------- Producer ----------
    def put(self, item: Any) -> bool:
        try:
            self._q.put_nowait(item)
        except Full:
            try:                     # bỏ phần tử cũ nhất
                self._q.get_nowait()
            except Empty:
                pass
            self._q.put_nowait(item)
        return True

    # ---------- Consumer ----------
    def get(self, timeout: Optional[float] = None) -> Any | None:
        try:
            return self._q.get(timeout=timeout)
        except Empty:
            return None

    # ---------- Helpers ----------
    def empty(self) -> bool:
        try:
            return self._q.empty()
        except NotImplementedError:
            return False

    def full(self) -> bool:
        try:
            return self._q.full()
        except NotImplementedError:
            return False

    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except NotImplementedError:
            return 0

    def clear(self) -> None:
        while self.get(timeout=0) is not None:
            pass

