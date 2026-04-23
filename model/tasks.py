from dataclasses import dataclass


@dataclass
class VideoTask:
    input_path: str
    output_path: str
    source_path: str
    start_frame: int = 0
    num_frames: int = 0
