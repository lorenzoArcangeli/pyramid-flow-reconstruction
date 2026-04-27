from dataclasses import dataclass


@dataclass
class VideoTask:
    input_path: str
    output_path: str
    source_path: str
