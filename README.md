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

## Dependencies

Install the Python packages listed in:

```bash
pip install -r requirements.txt
```

You also need:

- `ffmpeg` available in `PATH`
