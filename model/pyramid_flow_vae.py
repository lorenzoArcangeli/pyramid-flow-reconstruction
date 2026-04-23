import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import DEFAULT_CHUNK_FRAMES


def _next_valid_chunk(n: int) -> int:
    """Return the smallest T >= n such that (T-1) % 8 == 0."""
    if n <= 1:
        return 9
    remainder = (n - 1) % 8
    return n if remainder == 0 else n + (8 - remainder)


def _auto_chunk_frames_for_device(device: str) -> int:
    """Pick a VRAM-aware chunk size. Returned value already satisfies (N-1)%8==0."""
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return 17
    try:
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    except Exception:
        return 17

    if total_gb >= 20:
        return 65
    if total_gb >= 12:
        return 33
    return 17


class PyramidFlowVAEWrapper(nn.Module):
    def __init__(
        self,
        pyramid_flow_repo: str,
        vae_checkpoint: str,
        device: str = "cuda:0",
        use_fp16: bool = True,
        tile_sample_min_size: int = 256,
        chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    ):
        super().__init__()
        self.device = device
        self.use_fp16 = use_fp16
        self.encode_window_size = 8
        self.decode_window_size = 2
        self.tile_sample_min_size = tile_sample_min_size
        if chunk_frames <= 0:
            chunk_frames = _auto_chunk_frames_for_device(device)
        self.chunk_frames = _next_valid_chunk(chunk_frames)

        if pyramid_flow_repo not in sys.path:
            sys.path.insert(0, pyramid_flow_repo)

        from video_vae import CausalVideoVAE

        self.vae = CausalVideoVAE.from_pretrained(vae_checkpoint)
        dtype = torch.bfloat16 if use_fp16 else torch.float32
        self.vae = self.vae.to(dtype=dtype, device=device)
        self.vae.eval()
        self.vae.enable_tiling()

        if hasattr(torch, "compile"):
            try:
                self.vae = torch.compile(self.vae, mode="reduce-overhead", fullgraph=False)
            except Exception:
                pass

    def _process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        real_frames = chunk.shape[2]
        target_frames = _next_valid_chunk(real_frames)

        x = (chunk * 2.0) - 1.0
        dtype = torch.bfloat16 if self.use_fp16 else torch.float32
        x = x.to(dtype=dtype, device=self.device, non_blocking=True)

        if target_frames != real_frames:
            pad = target_frames - real_frames
            x = torch.cat([x, x[:, :, -1:, :, :].expand(-1, -1, pad, -1, -1)], dim=2)

        amp_ctx = torch.amp.autocast("cuda", enabled=self.use_fp16, dtype=torch.bfloat16)
        with torch.inference_mode(), amp_ctx:
            posterior = self.vae.encode(
                x,
                temporal_chunk=True,
                window_size=self.encode_window_size,
                tile_sample_min_size=self.tile_sample_min_size,
            ).latent_dist
            latents = posterior.mode()
            del x, posterior
            decoded = self.vae.decode(
                latents,
                temporal_chunk=True,
                window_size=self.decode_window_size,
                tile_sample_min_size=self.tile_sample_min_size,
            )
            del latents
            x_out = decoded.sample if hasattr(decoded, "sample") else decoded
            x_out = x_out[:, :, :real_frames].clamp(-1.0, 1.0)

        x_out = ((x_out + 1.0) / 2.0).to(dtype=torch.float32, device="cpu")
        return x_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, total_frames, h, w = x.shape
        chunk_size = self.chunk_frames

        if total_frames <= chunk_size:
            out = self._process_chunk(x)
            if out.shape[2:] != (total_frames, h, w):
                _, _, t_out, h_out, w_out = out.shape
                out = F.interpolate(
                    out.view(1, c * t_out, h_out, w_out),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).view(1, c, t_out, h, w)
            return out

        result = torch.empty((1, c, total_frames, h, w), dtype=torch.float32, device="cpu")
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunk = x[:, :, start:end].contiguous()
            out_chunk = self._process_chunk(chunk)
            result[:, :, start:end].copy_(out_chunk)
            del chunk, out_chunk

        if result.shape[3:] != (h, w):
            _, _, _, h_out, w_out = result.shape
            result = F.interpolate(
                result.squeeze(0).view(c * total_frames, 1, h_out, w_out),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).view(1, c, total_frames, h, w)

        return result
