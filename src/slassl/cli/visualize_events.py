from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from imageio_ffmpeg import write_frames
from PIL import Image

from slassl.data.h5_events import H5EventReader


def render_events(
    x: np.ndarray,
    y: np.ndarray,
    polarity: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid].astype(np.int64, copy=False)
    y = y[valid].astype(np.int64, copy=False)
    polarity = polarity[valid] > 0
    negative = np.zeros((height, width), dtype=np.float32)
    positive = np.zeros((height, width), dtype=np.float32)
    np.add.at(negative, (y[~polarity], x[~polarity]), 1)
    np.add.at(positive, (y[polarity], x[polarity]), 1)

    magnitude = np.maximum(negative, positive)
    nonzero = magnitude[magnitude > 0]
    scale = float(np.quantile(np.log1p(nonzero), 0.99)) if nonzero.size else 1.0
    negative = np.clip(np.log1p(negative) / max(scale, 1e-6), 0.0, 1.0)
    positive = np.clip(np.log1p(positive) / max(scale, 1e-6), 0.0, 1.0)

    image = np.full((height, width, 3), 20.0, dtype=np.float32)
    image[..., 0] += 235.0 * negative + 44.0 * positive
    image[..., 1] += 44.0 * negative + 140.0 * positive
    image[..., 2] += 44.0 * negative + 235.0 * positive
    return np.clip(image, 0, 255).astype(np.uint8)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Render event windows from a native HDF5 file")
    result.add_argument("--input", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--camera", choices=["left", "right"], default="left")
    result.add_argument("--event-group")
    result.add_argument("--timestamp-scale-to-us", type=float)
    result.add_argument("--timestamp-offset-us", type=float)
    result.add_argument("--timestamp-us", type=int, help="End timestamp of the first frame")
    result.add_argument("--window-us", type=int, default=5000)
    result.add_argument("--step-us", type=int, default=5000)
    result.add_argument("--frames", type=int, default=1)
    result.add_argument("--fps", type=float, default=20.0)
    result.add_argument("--height", type=int, default=260)
    result.add_argument("--width", type=int, default=346)
    return result


def save_mp4(frames: list[Image.Image], output: Path, fps: float) -> None:
    width, height = frames[0].size
    writer = write_frames(
        str(output),
        (width, height),
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        fps=fps,
        codec="libx264",
        macro_block_size=2,
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


def main() -> None:
    args = parser().parse_args()
    if args.window_us <= 0 or args.step_us <= 0 or args.frames <= 0 or args.fps <= 0:
        raise ValueError("window-us, step-us, frames, and fps must be positive")
    frames: list[Image.Image] = []
    frame_metadata = []
    with H5EventReader(
        args.input,
        camera=args.camera,
        event_group=args.event_group,
        timestamp_scale_to_us=args.timestamp_scale_to_us,
        timestamp_offset_us=args.timestamp_offset_us,
    ) as reader:
        first_us, last_us = reader.bounds_us()
        if args.timestamp_us is not None:
            first_end_us = args.timestamp_us
        elif args.frames == 1:
            first_end_us = (first_us + last_us) // 2
        else:
            first_end_us = first_us + args.window_us
        for index in range(args.frames):
            end_us = first_end_us + index * args.step_us
            if end_us > last_us:
                break
            start_us = end_us - args.window_us
            x, y, _, polarity = reader.read_us(start_us, end_us)
            frames.append(Image.fromarray(render_events(x, y, polarity, args.height, args.width)))
            frame_metadata.append(
                {"start_us": start_us, "end_us": end_us, "events": int(len(x))}
            )
        layout = reader.describe()
    if not frames:
        raise ValueError("No frames were rendered within the recording time range")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower()
    if suffix == ".mp4":
        save_mp4(frames, args.output, args.fps)
    elif len(frames) == 1:
        frames[0].save(args.output)
    else:
        if suffix != ".gif":
            raise ValueError("Multiple frames require an output path ending in .gif or .mp4")
        frames[0].save(
            args.output,
            save_all=True,
            append_images=frames[1:],
            duration=max(round(1000.0 / args.fps), 1),
            loop=0,
        )
    print(
        json.dumps(
            {"output": str(args.output), "layout": layout, "frames": frame_metadata},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
