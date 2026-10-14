"""Resume the interrupted full render from the preserved D: frame cache."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from replace_background import (
    MODEL_NAME, WORKERS, SUBJECT_SCALE, SUBJECT_BOTTOM_MARGIN,
    BACKGROUND_BLUR_SIGMA, init_worker, remove_background,
    clean_matte_sequence, ffmpeg_path, run,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "input" / "M345VI_1351.MP4"
BACKGROUND = ROOT / "input" / "ChatGPT Image Sep 1, 2026, 09_41_22 PM (2).png"
TEMP = ROOT / "temp" / "background_replace_dz1te_26"
FRAMES = TEMP / "frames"
KEYED = TEMP / "keyed"
OUTPUT = ROOT / "output" / "M345VI_1351_NEW_BACKGROUND_FULL.mp4"
ANALYSIS_FPS = 18


def main() -> int:
    frame_paths = sorted(FRAMES.glob("*.png"))
    bad = []
    for path in sorted(KEYED.glob("*.png")):
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            bad.append(path)
            path.unlink(missing_ok=True)
    if bad:
        print(f"Removed {len(bad)} incomplete keyed frame(s); regenerating them.", flush=True)
    keyed_names = {path.name for path in KEYED.glob("*.png")}
    missing = [path for path in frame_paths if path.name not in keyed_names]
    print(f"Saved cache found: {len(frame_paths)} frames; {len(missing)} remaining.", flush=True)
    if missing:
        tasks = [(str(path), str(KEYED / path.name)) for path in missing]
        with ProcessPoolExecutor(
            max_workers=WORKERS, initializer=init_worker, initargs=(MODEL_NAME,)
        ) as pool:
            for index, _ in enumerate(pool.map(remove_background, tasks), 1):
                if index % 30 == 0 or index == len(tasks):
                    print(f"Resumed {index}/{len(tasks)} frames", flush=True)

    print("Cleaning and stabilizing the complete matte...", flush=True)
    clean_matte_sequence(sorted(KEYED.glob("*.png")))

    subject_width = int(1920 * SUBJECT_SCALE)
    subject_height = int(1080 * SUBJECT_SCALE)
    subject_x = (1920 - subject_width) // 2
    subject_y = 1080 - subject_height - SUBJECT_BOTTOM_MARGIN
    filter_graph = (
        f"[1:v]scale=1920:1080,gblur=sigma={BACKGROUND_BLUR_SIGMA}:steps=2[bg];"
        f"[0:v]scale={subject_width}:{subject_height}:flags=lanczos,"
        "eq=brightness=-0.035:contrast=1.03:saturation=0.96,"
        "colorbalance=rs=-0.025:gs=0.01:bs=0.03:rm=0.03:gm=0.01:bm=-0.015[fg];"
        f"[bg][fg]overlay={subject_x}:{subject_y}:format=auto,"
        "noise=alls=2:allf=t+u,format=yuv420p[v]"
    )
    print("Compositing the resumed full video...", flush=True)
    run([
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(ANALYSIS_FPS), "-i", str(KEYED / "%06d.png"),
        "-loop", "1", "-i", str(BACKGROUND), "-i", str(SOURCE),
        "-filter_complex", filter_graph, "-map", "[v]", "-map", "2:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-ar", "48000", "-ac", "2", "-r", "30", "-shortest",
        "-movflags", "+faststart", str(OUTPUT),
    ])
    print(f"Done: {OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
