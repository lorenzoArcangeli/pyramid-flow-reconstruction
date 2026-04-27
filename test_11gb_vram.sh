#!/bin/bash

# Configuration to run video reconstruction on a GPU with 11 GB VRAM

# Setting memory allocator to minimize memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Use local ffmpeg if applicable, or remove this if you use system ffmpeg
# export PATH="/media/SSD_4TB/Trends_2025/Project_3/local-ffmpeg:$PATH"

INPUT_FOLDER="/media/SSD_4TB/SAFE-video-challenge/small_dataset/videodiffusion/clips_original/640x360_23/"
OUTPUT_ROOT="test"

mkdir -p "$OUTPUT_ROOT"

python -m prepare_reconstruction_AIGVDBench \
  --input-folder "$INPUT_FOLDER" \
  --output-root "$OUTPUT_ROOT" \
  --devices "cuda:0" \
  --limit 2 \
  --target-width 384 \
  --target-height 384 \
  --chunk-frames 9 \
  --tile-sample-min-size 256 \
  --aggressive-memory-cleanup