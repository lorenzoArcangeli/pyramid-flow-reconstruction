import queue
import threading

from .constants import PREFETCH_QUEUE_SIZE
from .video_io import read_video_cpu


class VideoPrefetcher:
    """Background thread that pre-loads videos to CPU pinned memory."""

    def __init__(self, tasks, args, maxsize: int = PREFETCH_QUEUE_SIZE):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(
            target=self._load_loop, args=(tasks, args), daemon=True
        )
        self._thread.start()

    def _load_loop(self, tasks, args):
        for task in tasks:
            try:
                tensor, fps, nf = read_video_cpu(
                    task.input_path,
                    target_width=args.target_width,
                    target_height=args.target_height,
                    max_frames=args.max_frames,
                    decode_threads=args.decode_threads,
                )
                self._q.put((task, tensor, fps, nf, None))
            except Exception as exc:
                self._q.put((task, None, None, None, exc))
        self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item
