# recontruct_video

Small utility repo to reconstruct a folder of videos with the Pyramid Flow VAE.

`ffmpeg` must be installed and available in your shell `PATH`.

## Main script

Use:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python prepare_reconstruction_AIGVDBench.py \
  --input-folder /path/to/input_videos \
  --output-root /path/to/output_videos
```

Only `--input-folder` and `--output-root` are required.

## Current defaults

These defaults are already set inside the script:

```text
--pyramid-flow-repo <repo-root>/Pyramid-Flow
--vae-checkpoint <repo-root>/pyramid-flow-miniflux/causal_video_vae
--devices cuda:0
--skip-existing
--max-frames 64
--prefetch-queue-size 4
--decode-threads 8
--ffmpeg-preset veryfast
--ffmpeg-threads 8
```

If needed, any of them can still be overridden from the command line.


## Output behavior

- Reconstructed videos are written into `--output-root`.
- The reconstructed video filename is exactly the same as the real input video filename.
- Existing outputs are skipped by default. Use `--no-skip-existing` to rebuild them.

## Single-GPU (RTX 4090, 24 GB)

The RTX 4090 has 24 GB VRAM. Use `--chunk-frames 33` to keep each temporal chunk
within budget (33 frames ≈ 12–14 GB peak). You can run directly without `accelerate`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python prepare_reconstruction_AIGVDBench.py \
  --devices cuda:0 \
  --chunk-frames 33 \
  --input-folder /path/to/input_videos \
  --output-root /path/to/output_videos
```

Or via `accelerate` (matches the style of the multi-GPU commands below):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
  --config_file accelerate_config_1gpu.yaml \
  prepare_reconstruction_AIGVDBench.py \
  --devices cuda:0 \
  --chunk-frames 33 \
  --input-folder /path/to/input_videos \
  --output-root /path/to/output_videos
```

---

## Multi-GPU parallelism (4× NVIDIA A40)

Launch with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
  --config_file accelerate_config_4gpu.yaml \
  prepare_reconstruction_AIGVDBench.py \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --input-folder /path/to/input_videos \
  --output-root /path/to/output_videos
```

`accelerate launch` spawns one process per GPU, setting `WORLD_SIZE=4` and
`LOCAL_RANK=0..3`. Each rank automatically receives a disjoint round-robin
slice of the dataset (`rank::world_size`) and loads the VAE onto its own GPU,
so all four cards run fully independently with no inter-GPU communication.

### 8-GPU variant

Use [accelerate_config_8gpu.yaml](accelerate_config_8gpu.yaml):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
  --config_file accelerate_config_8gpu.yaml \
  prepare_reconstruction_AIGVDBench.py \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
  --input-folder /path/to/input_videos \
  --output-root /path/to/output_videos
```

Or with `torchrun`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=8 \
  prepare_reconstruction_AIGVDBench.py \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
  --input-folder /path/to/input_videos \
  --output-root /path/to/output_videos
```

### 4-GPU — torchrun alternative

If the cluster uses `torchrun` instead of `accelerate`, the command is equivalent:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=4 \
  prepare_reconstruction_AIGVDBench.py \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --input-folder /path/to/input_videos \
  --output-root /path/to/output_videos
```

## Dependencies

Install the Python packages listed in:

```bash
pip install -r requirements.txt
```

You also need:

- `ffmpeg` available in `PATH`
