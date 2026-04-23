import os
import subprocess

import av
import cv2
import numpy as np
import torch


def read_video_cpu(
    video_path: str,
    target_width: int | None,
    target_height: int | None,
    max_frames: int,
    decode_threads: int = 0,
) -> tuple[torch.Tensor, float, int]:
    """Decode video on CPU and return a pinned-memory float32 tensor."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    if decode_threads > 0:
        stream.thread_count = decode_threads
    fps = float(stream.average_rate) if stream.average_rate is not None else 30.0

    frames: list[np.ndarray] = []
    resize_w: int | None = None
    resize_h: int | None = None

    for i, frame in enumerate(container.decode(video=0)):
        if max_frames > 0 and i >= max_frames:
            break
        img = frame.to_ndarray(format="rgb24")
        h, w = img.shape[0], img.shape[1]

        if target_width and target_height:
            if w != target_width or h != target_height:
                img = cv2.resize(
                    img,
                    (target_width, target_height),
                    interpolation=cv2.INTER_LINEAR,
                )
        else:
            if resize_w is None or resize_h is None:
                short_edge = min(h, w)
                if short_edge > 0:
                    scale = 640.0 / float(short_edge)
                    resize_w = int(round(w * scale))
                    resize_h = int(round(h * scale))
                else:
                    resize_w, resize_h = w, h

            if resize_w is not None and resize_h is not None:
                if w != resize_w or h != resize_h:
                    img = cv2.resize(
                        img,
                        (resize_w, resize_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
        frames.append(img)

    container.close()

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")

    arr = np.stack(frames, axis=0)
    tensor = torch.from_numpy(arr).float().div_(255.0)
    tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
    tensor = tensor.pin_memory()
    return tensor, fps, len(frames)


def compress_and_save_video(
    video_tensor: torch.Tensor,
    output_path: str,
    fps: float,
    ffmpeg_preset: str = "veryfast",
    ffmpeg_threads: int = 0,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    t = video_tensor
    if t.shape[1] == 3:
        t = t.squeeze(0).permute(1, 2, 3, 0)
    else:
        t = t.squeeze(0).permute(0, 2, 3, 1)

    frames_np = (t.cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
    _, h, w, _ = frames_np.shape

    if h % 2 or w % 2:
        h2, w2 = h + h % 2, w + w % 2
        frames_np = np.pad(frames_np, ((0, 0), (0, h2 - h), (0, w2 - w), (0, 0)), mode="edge")
        h, w = h2, w2

    crf = int(np.random.randint(16, 31))
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vf", "format=yuv420p",
        "-c:v", "libx264",
        "-preset", ffmpeg_preset,
        "-profile:v", "main",
        "-level", "3.1",
        "-crf", str(crf),
        output_path,
    ]
    if ffmpeg_threads > 0:
        ffmpeg_cmd[1:1] = ["-threads", str(ffmpeg_threads)]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    proc.stdin.write(frames_np.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (code {proc.returncode}) for {output_path}")
