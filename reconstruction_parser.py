import argparse

from model import DEFAULT_CHUNK_FRAMES

from reconstruction_constants import (
    DEFAULT_DECODE_THREADS,
    DEFAULT_DEVICES,
    DEFAULT_FFMPEG_PRESET,
    DEFAULT_FFMPEG_THREADS,
    DEFAULT_MAX_FRAMES,
    DEFAULT_PREFETCH_QUEUE_SIZE,
    DEFAULT_PYRAMID_FLOW_REPO,
    DEFAULT_VAE_CHECKPOINT,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Pyramid Flow VAE reconstructions (optimized)."
    )
    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pyramid-flow-repo", default=DEFAULT_PYRAMID_FLOW_REPO)
    parser.add_argument("--vae-checkpoint", default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--devices", default=DEFAULT_DEVICES)
    parser.add_argument("--target-width", type=int, default=None)
    parser.add_argument("--target-height", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--tile-sample-min-size", type=int, default=256)
    parser.add_argument("--force-fps", type=float, default=None)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prefetch-queue-size",
        type=int,
        default=DEFAULT_PREFETCH_QUEUE_SIZE,
        help="How many decoded videos to keep queued on CPU.",
    )
    parser.add_argument(
        "--decode-threads",
        type=int,
        default=DEFAULT_DECODE_THREADS,
        help="PyAV decode threads per video stream (0 = ffmpeg auto).",
    )
    parser.add_argument(
        "--ffmpeg-preset",
        default=DEFAULT_FFMPEG_PRESET,
        help="x264 preset for output encoding (ultrafast..placebo).",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=DEFAULT_FFMPEG_THREADS,
        help="ffmpeg encode threads (0 = ffmpeg auto).",
    )
    parser.add_argument(
        "--aggressive-memory-cleanup",
        action="store_true",
        help="Call gc + cuda empty_cache after each video/chunk (safer, but slower).",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=DEFAULT_CHUNK_FRAMES,
        help="Frames per VAE temporal chunk. Must satisfy (N-1)%%8==0. "
        "Use 0 for automatic VRAM-based selection (default).",
    )
    return parser.parse_args()
