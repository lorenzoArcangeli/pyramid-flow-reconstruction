import queue
import threading

from .video_io import read_video_cpu

_QUEUE_PUT_TIMEOUT = 1  # seconds; allows the shutdown event to be checked regularly


class VideoPrefetcher:
    """Background thread that pre-loads videos to CPU pinned memory."""

    def __init__(self, tasks, args, maxsize: int = 2):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._load_loop, args=(tasks, args), daemon=True
        )
        self._thread.start()

    def _load_loop(self, tasks, args):
        for task in tasks:
            if self._stop.is_set():
                return
            try:
                tensor, fps, nf = read_video_cpu(
                    task.input_path,
                    target_width=args.target_width,
                    target_height=args.target_height,
                    max_frames=args.max_frames,
                    decode_threads=args.decode_threads,
                )
                item = (task, tensor, fps, nf, None)
            except Exception as exc:
                item = (task, None, None, None, exc)

            while not self._stop.is_set():
                try:
                    self._q.put(item, timeout=_QUEUE_PUT_TIMEOUT)
                    break
                except queue.Full:
                    continue

        if not self._stop.is_set():
            # Sentinel — may also block if queue is full and consumer stopped.
            while not self._stop.is_set():
                try:
                    self._q.put(None, timeout=_QUEUE_PUT_TIMEOUT)
                    break
                except queue.Full:
                    continue

    def close(self):
        """Signal the loader thread to stop and wait for it to exit."""
        self._stop.set()
        self._thread.join()

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
