import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> tuple[bool, int]:
        now = time.time()
        queue = self._buckets[key]

        while queue and now - queue[0] > self.window_seconds:
            queue.popleft()

        if len(queue) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - queue[0]))
            return False, max(retry_after, 1)

        queue.append(now)
        return True, 0
