import gc
import os
import queue
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from model import (
    VIDEO_EXTENSIONS,
    PyramidFlowVAEWrapper,
    VideoPrefetcher,
    VideoTask,
    compress_and_save_video,
)
from reconstruction_constants import (
    DEFAULT_VAE_CHECKPOINT_DIRNAME,
    DEFAULT_VAE_MODEL_REPO,
    DEFAULT_VAE_REQUIRED_FILE,
)
from reconstruction_parser import parse_args

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

"""
cd /media/SSD_4TB/Trends_2025/Project_3/Pitaya-videodet/src

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PATH="/media/SSD_4TB/Trends_2025/Project_3/local-ffmpeg:$PATH" \
python -m prepare_reconstruction_AIGVDBench \
  --input-folder /media/SSD_4TB/SAFE-video-challenge/small_dataset/videos \
  --output-root /media/SSD_4TB/Trends_2025/Project_3/recontructed_video
"""

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def collect_paths(input_folder: str) -> list[str]:
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder does not exist or is not a directory: {input_folder}")

    paths = []
    for entry in os.scandir(input_folder):
        if not entry.is_file():
            continue
        if os.path.splitext(entry.name)[1].lower() not in VIDEO_EXTENSIONS:
            continue
        paths.append(os.path.abspath(entry.path))

    paths.sort()
    if not paths:
        raise ValueError(f"No supported video files found in input folder: {input_folder}")
    return paths


def is_valid_vae_checkpoint(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(
        os.path.join(path, DEFAULT_VAE_REQUIRED_FILE)
    )


def ensure_vae_checkpoint_available(args) -> None:
    requested_path = os.path.abspath(args.vae_checkpoint)
    candidate_paths = [requested_path]

    if os.path.basename(requested_path) != DEFAULT_VAE_CHECKPOINT_DIRNAME:
        candidate_paths.append(
            os.path.join(requested_path, DEFAULT_VAE_CHECKPOINT_DIRNAME)
        )

    for candidate in candidate_paths:
        if is_valid_vae_checkpoint(candidate):
            args.vae_checkpoint = candidate
            print(f"[INFO] Using VAE checkpoint: {args.vae_checkpoint}")
            return

    download_root = (
        os.path.dirname(requested_path)
        if os.path.basename(requested_path) == DEFAULT_VAE_CHECKPOINT_DIRNAME
        else requested_path
    )
    os.makedirs(download_root, exist_ok=True)

    print(
        "[INFO] VAE checkpoint not found. Downloading "
        f"{DEFAULT_VAE_MODEL_REPO} to {download_root} ..."
    )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=DEFAULT_VAE_MODEL_REPO,
        local_dir=download_root,
        local_dir_use_symlinks=False,
        repo_type="model",
    )

    downloaded_checkpoint = os.path.join(
        download_root,
        DEFAULT_VAE_CHECKPOINT_DIRNAME,
    )
    if not is_valid_vae_checkpoint(downloaded_checkpoint):
        raise FileNotFoundError(
            "Downloaded model snapshot is missing the expected VAE checkpoint folder: "
            f"{downloaded_checkpoint}"
        )

    args.vae_checkpoint = downloaded_checkpoint
    print(f"[INFO] Using downloaded VAE checkpoint: {args.vae_checkpoint}")


def sample_paths(paths: list[str], sample_ratio: float, sample_seed: int) -> list[str]:
    if sample_ratio >= 1.0:
        return paths
    if sample_ratio <= 0.0:
        return []
    count = max(1, int(round(len(paths) * sample_ratio)))
    rng = np.random.default_rng(sample_seed)
    idx = sorted(rng.choice(len(paths), size=count, replace=False).tolist())
    return [paths[i] for i in idx]


def make_output_path(input_path: str, output_root: str) -> str:
    # Keep the reconstructed filename identical to the source filename.
    return os.path.join(output_root, os.path.basename(os.path.abspath(input_path)))


def ensure_matching_video_names(real_path: str, reconstructed_path: str) -> None:
    real_name = os.path.basename(os.path.abspath(real_path))
    reconstructed_name = os.path.basename(os.path.abspath(reconstructed_path))
    if real_name != reconstructed_name:
        raise ValueError(
            "The reconstructed video name must match the real video name: "
            f"{real_name} != {reconstructed_name}"
        )


# ---------------------------------------------------------------------------
# VAE wrapper – now with temporal chunking to bound VRAM
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(
    task: VideoTask,
    tensor: torch.Tensor,
    fps: float,
    num_frames: int,
    vae: PyramidFlowVAEWrapper,
    args,
):
    # Non-blocking GPU transfer (tensor is already pinned)
    reconstruction = vae(tensor)
    # vae returns (1, C, T, H, W); compress_and_save expects same
    compress_and_save_video(
        reconstruction,
        task.output_path,
        args.force_fps or fps,
        ffmpeg_preset=args.ffmpeg_preset,
        ffmpeg_threads=args.ffmpeg_threads,
    )
    del tensor, reconstruction


# ---------------------------------------------------------------------------
# Worker (one per GPU device)
# ---------------------------------------------------------------------------

def _emit_progress_event(progress_queue, event: dict):
    if progress_queue is None:
        return
    try:
        progress_queue.put_nowait(event)
    except Exception:
        # Never block processing due to telemetry issues.
        pass


def worker_process(
    device: str,
    tasks: list[VideoTask],
    args,
    progress_queue=None,
):
    vae = PyramidFlowVAEWrapper(
        pyramid_flow_repo=args.pyramid_flow_repo,
        vae_checkpoint=args.vae_checkpoint,
        device=device,
        use_fp16=True,
        tile_sample_min_size=args.tile_sample_min_size,
        chunk_frames=args.chunk_frames,
    )

    prefetcher = VideoPrefetcher(tasks, args, maxsize=max(1, args.prefetch_queue_size))

    for item in tqdm(prefetcher, total=len(tasks), desc=device, position=0, leave=True):
        task, tensor, fps, num_frames, exc = item
        t0 = time.perf_counter()
        if exc is not None:
            print(f"[{device}] Load error {task.input_path}: {exc}")
            _emit_progress_event(
                progress_queue,
                {
                    "event": "video_done",
                    "status": "load_error",
                    "device": device,
                    "input_path": task.input_path,
                    "output_path": task.output_path,
                    "num_frames": 0,
                    "duration_sec": time.perf_counter() - t0,
                    "error": str(exc),
                },
            )
            continue
        try:
            process_video(task, tensor, fps, num_frames, vae, args)
            _emit_progress_event(
                progress_queue,
                {
                    "event": "video_done",
                    "status": "success",
                    "device": device,
                    "input_path": task.input_path,
                    "output_path": task.output_path,
                    "num_frames": int(num_frames),
                    "duration_sec": time.perf_counter() - t0,
                    "error": "",
                },
            )
        except Exception as e:
            print(f"[{device}] Process error {task.input_path}: {e}")
            _emit_progress_event(
                progress_queue,
                {
                    "event": "video_done",
                    "status": "process_error",
                    "device": device,
                    "input_path": task.input_path,
                    "output_path": task.output_path,
                    "num_frames": int(num_frames),
                    "duration_sec": time.perf_counter() - t0,
                    "error": str(e),
                },
            )
        finally:
            if args.aggressive_memory_cleanup:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Task distribution
# ---------------------------------------------------------------------------

def chunk_tasks(tasks: list[VideoTask], devices: list[str]) -> list[list[VideoTask]]:
    buckets: list[list[VideoTask]] = [[] for _ in devices]
    for i, task in enumerate(tasks):
        buckets[i % len(devices)].append(task)
    return buckets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _drain_progress_events(progress_queue, stats: dict):
    drained = 0
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            break
        drained += 1
        if event.get("event") != "video_done":
            continue

        status = event.get("status", "unknown")
        if status == "success":
            stats["success"] += 1
            stats["frames_ok"] += int(event.get("num_frames", 0))
            stats["seconds_ok"] += float(event.get("duration_sec", 0.0))
        elif status == "load_error":
            stats["load_error"] += 1
        else:
            stats["process_error"] += 1
        stats["processed"] += 1
        stats["last_error"] = event.get("error", "")
        stats["last_device"] = event.get("device", "")
        stats["last_output_path"] = event.get("output_path", "")
    return drained


def _monitor_workers(processes: list[mp.Process], progress_queue):
    stats = {
        "processed": 0,
        "success": 0,
        "load_error": 0,
        "process_error": 0,
        "frames_ok": 0,
        "seconds_ok": 0.0,
        "last_error": "",
        "last_device": "",
        "last_output_path": "",
    }

    while True:
        _drain_progress_events(progress_queue, stats)
        alive = any(p.is_alive() for p in processes)
        if not alive:
            _drain_progress_events(progress_queue, stats)
            break
        time.sleep(0.5)
    return stats


def main():
    args = parse_args()
    ensure_vae_checkpoint_available(args)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    source_paths = collect_paths(args.input_folder)
    source_paths = sample_paths(source_paths, args.sample_ratio, args.sample_seed)
    if args.limit > 0:
        source_paths = source_paths[: args.limit]

    tasks = [
        VideoTask(
            input_path=p,
            output_path=make_output_path(p, args.output_root),
            source_path=p,
        )
        for p in source_paths
    ]
    for task in tasks:
        ensure_matching_video_names(task.source_path, task.output_path)
    if args.skip_existing:
        tasks = [t for t in tasks if not os.path.exists(t.output_path)]

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        raise ValueError("No devices provided")

    buckets = chunk_tasks(tasks, devices)
    processes = []
    progress_queue = mp.Queue(maxsize=4096)
    for device, bucket in zip(devices, buckets):
        if not bucket:
            continue
        p = mp.Process(
            target=worker_process,
            args=(device, bucket, args, progress_queue),
        )
        p.start()
        processes.append(p)

    _monitor_workers(processes, progress_queue)
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
