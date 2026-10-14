"""Reusable video background replacer for Windows.

Put one video and one background image in the input folder, edit the three
settings below, then run this file. The result is written to output/.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps
from rembg import new_session, remove
import torch


# ========================= OPTIONAL SETTINGS ===========================
# Leave these blank. The script will automatically find the files you put
# in input/. If there is more than one, it will ask you to choose by number.
VIDEO_NAME = ""
BACKGROUND_NAME = ""
OUTPUT_NAME = ""

# Balanced speed/accuracy mode:
# The human-specific model is the most reliable model that fits this laptop's
# memory. 12 analysis FPS and 768x432 keep the CPU workload manageable; the
# final composite is still exported at 30 FPS.
PROCESSING_FPS = 12
SEGMENTATION_SIZE = (768, 432)
SUBJECT_SCALE = 0.86
SUBJECT_BOTTOM_MARGIN = 0
# Put a safety border around the source before matting.  This prevents hair
# or hands touching the camera-frame boundary from becoming a hard rectangle.
SOURCE_PAD_SCALE = 0.90
BACKGROUND_BLUR_SIGMA = 5
# RVM is a recurrent video matting model: frames are processed sequentially
# and its memory is carried forward, preventing hand/ear flicker and halos.
MODEL_NAME = "rvm_mobilenetv3"
WORKERS = 2
# Apply one-pixel edge contraction during final cleanup.  This removes the
# bright fringe without the earlier double-contraction that damaged hair,
# ears, and hands.
EDGE_EROSION_SIZE = 3
EDGE_FEATHER_RADIUS = 0.15
ALPHA_HARDEN_THRESHOLD = 64
TEMPORAL_ALPHA_BLEND = 0.0
SHADOW_OPACITY = 46
SHADOW_BLUR_RADIUS = 18
SHADOW_Y_OFFSET = 10
LIGHT_WRAP_OPACITY = 0
LIGHT_WRAP_COLOR = (218, 158, 88)
# =======================================================================

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
TEMP_DIR = ROOT / "temp"
LOG_FILE = ROOT / "last_run.log"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

_SESSION = None
_RVM_MODEL = None


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("FFmpeg was not found on PATH.")
    return path


def ffprobe_path() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise RuntimeError("ffprobe was not found on PATH.")
    return path


def init_worker(model: str) -> None:
    global _SESSION
    _SESSION = new_session(model)


def load_rvm_model():
    global _RVM_MODEL
    if _RVM_MODEL is None:
        vendor = ROOT / "vendor" / "RobustVideoMatting"
        _RVM_MODEL = torch.hub.load(str(vendor), "mobilenetv3", source="local")
        _RVM_MODEL = _RVM_MODEL.eval().cpu()
    return _RVM_MODEL


def process_rvm_sequence(frame_paths: list[Path], keyed: Path) -> None:
    """Create temporally stable RGBA frames with RVM recurrent memory."""
    model = load_rvm_model()
    rec = [None] * 4
    with torch.no_grad():
        for index, frame_path in enumerate(frame_paths, 1):
            rgb = Image.open(frame_path).convert("RGB")
            array = np.asarray(rgb, dtype=np.float32) / 255.0
            source = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
            fgr, pha, *rec = model(source, *rec, 0.5)
            alpha = (pha[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
            # RVM's soft edge can retain bright source-background spill. Keep
            # the opaque subject, but remove weak fringe before compositing.
            alpha_image = Image.fromarray(alpha, mode="L")
            # RVM already returns a temporally stable soft matte.  Do not
            # erode it here: a second erosion destroys the top of the head,
            # ears, and moving fingers.  Only remove very weak fringe pixels.
            alpha_image = alpha_image.point(
                lambda value: 0 if value < ALPHA_HARDEN_THRESHOLD else value
            )
            # Use RVM's reconstructed foreground instead of the original RGB
            # frame.  This removes bright source-background contamination at
            # the hair/shoulder boundary while preserving the soft matte.
            foreground = (
                fgr[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)
            cutout = Image.fromarray(foreground, mode="RGB").convert("RGBA")
            cutout.putalpha(alpha_image)
            # The padded canvas is only for giving RVM context around the
            # head. Remove that border before compositing, otherwise the
            # subject appears to float inside an internal rectangular frame.
            pad_w = int(SEGMENTATION_SIZE[0] * SOURCE_PAD_SCALE)
            pad_h = int(SEGMENTATION_SIZE[1] * SOURCE_PAD_SCALE)
            pad_x = (SEGMENTATION_SIZE[0] - pad_w) // 2
            pad_y = (SEGMENTATION_SIZE[1] - pad_h) // 2
            crop_box = (pad_x, pad_y, pad_x + pad_w, pad_y + pad_h)
            cutout = cutout.crop(crop_box).resize(
                SEGMENTATION_SIZE, Image.Resampling.LANCZOS
            )
            cutout.save(keyed / frame_path.name)
            if index % 30 == 0 or index == len(frame_paths):
                print(f"Processed {index}/{len(frame_paths)} frames", flush=True)


def remove_background(task: tuple[str, str]) -> str:
    source, target = task
    # Skip frames that were already processed (resume support)
    if Path(target).exists():
        return target
    image = Image.open(source).convert("RGBA")
    cutout = remove(
        image,
        session=_SESSION,
        alpha_matting=False,
        post_process_mask=True,
    )
    cutout.save(target)
    return target


def clean_matte_sequence(paths: list[Path]) -> None:
    """Reduce fringe and small frame-to-frame matte changes."""
    previous = None
    for path in paths:
        cutout = Image.open(path).convert("RGBA")
        alpha = cutout.getchannel("A")
        if EDGE_EROSION_SIZE >= 3:  # MinFilter requires size >= 3 in Pillow
            alpha = alpha.filter(ImageFilter.MinFilter(EDGE_EROSION_SIZE))
        alpha = alpha.filter(ImageFilter.GaussianBlur(EDGE_FEATHER_RADIUS))
        alpha = alpha.point(lambda value: 0 if value < ALPHA_HARDEN_THRESHOLD else value)
        if previous is not None:
            alpha = Image.blend(alpha, previous, TEMPORAL_ALPHA_BLEND)
        cutout.putalpha(alpha)

        shadow_blur = alpha.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_RADIUS))
        shifted_shadow = Image.new("L", alpha.size, 0)
        shifted_shadow.paste(shadow_blur, (0, SHADOW_Y_OFFSET))
        shadow_alpha = shifted_shadow.point(lambda value: value * SHADOW_OPACITY // 255)
        shadow = Image.new("RGBA", cutout.size, (7, 5, 4, 0))
        shadow.putalpha(shadow_alpha)

        wrap_source = alpha.filter(ImageFilter.GaussianBlur(5))
        wrap_alpha = ImageChops.subtract(wrap_source, alpha)
        wrap_alpha = wrap_alpha.point(lambda value: value * LIGHT_WRAP_OPACITY // 255)
        wrap = Image.new("RGBA", cutout.size, (*LIGHT_WRAP_COLOR, 0))
        wrap.putalpha(wrap_alpha)

        layer = Image.new("RGBA", cutout.size, (0, 0, 0, 0))
        layer.alpha_composite(shadow)
        layer.alpha_composite(wrap)
        layer.alpha_composite(cutout)
        layer.save(path)
        previous = alpha


def run(command: list[str], show_errors: bool = True) -> None:
    """Run a subprocess command, showing errors if it fails."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("\n[ERROR] Command failed:")
        print("  " + " ".join(str(c) for c in command))
        if result.stdout.strip():
            print("[stdout]", result.stdout.strip())
        if result.stderr.strip():
            print("[stderr]", result.stderr.strip())
        raise subprocess.CalledProcessError(result.returncode, command)


def choose_file(folder: Path, extensions: set[str], kind: str) -> Path:
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    if not files:
        raise FileNotFoundError(f"No {kind} found in {folder}")
    if len(files) == 1:
        return files[0]
    print(f"Choose the {kind}:")
    for index, path in enumerate(files, 1):
        print(f"  {index}. {path.name}")
    while True:
        answer = input(f"Enter the {kind} number (1-{len(files)}): ").strip()
        try:
            selected = files[int(answer) - 1]
        except (ValueError, IndexError):
            print("Please enter one of the numbers shown above.")
            continue
        return selected


def find_existing_temp(video_stem: str, expected_frames: int | None = None) -> Path | None:
    """Find an existing temp folder for this video to resume from.

    If expected_frames is given, only match a folder whose frame count is
    within 5% of that number (so a 30-second preview doesn't resume from a
    full-length folder or vice-versa).
    """
    marker = f"bg_{video_stem}_"
    candidates = []
    if TEMP_DIR.exists():
        for d in sorted(TEMP_DIR.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            frames = d / "frames"
            keyed = d / "keyed"
            if not (frames.exists() and keyed.exists()):
                continue
            frame_count = len(list(frames.glob("*.png")))
            keyed_count = len(list(keyed.glob("*.png")))
            if frame_count == 0:
                continue
            # If we know how many frames we expect, only resume if close match
            if expected_frames is not None:
                ratio = frame_count / expected_frames
                if not (0.90 <= ratio <= 1.10):
                    continue
            # Prefer folders made for this exact video. Older versions of the
            # script used generic names, so accept those only when the frame
            # count matches the probed video duration.
            is_exact_name = d.name.startswith(marker)
            # Never reuse masks from the old independent-frame backend when
            # running RVM. Its recurrent state makes those masks incompatible.
            if MODEL_NAME == "rvm_mobilenetv3" and not is_exact_name:
                continue
            candidates.append((is_exact_name, keyed_count, d))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2].stat().st_mtime), reverse=True)
        _, keyed_count, selected = candidates[0]
        frame_count = len(list((selected / "frames").glob("*.png")))
        print(f"[RESUME] Found existing work in: {selected.name}")
        print(f"         Frames extracted: {frame_count}, Keyed: {keyed_count}")
        return selected
    return None


def main() -> int:
    # Set up logging to file
    log = open(LOG_FILE, "w", encoding="utf-8")
    def log_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line, **kwargs)
        log.write(line + "\n")
        log.flush()

    log_print("=" * 60)
    log_print("Background Replacer started")
    log_print("=" * 60)

    try:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Optional absolute paths make it possible to process a large source
        # video in place without copying it into input/ or selecting a stale
        # trimmed file by mistake.
        video_setting = os.environ.get("BACKGROUND_VIDEO_PATH", "").strip() or VIDEO_NAME.strip()
        background_setting = os.environ.get("BACKGROUND_IMAGE_PATH", "").strip() or BACKGROUND_NAME.strip()
        source = Path(video_setting) if video_setting else choose_file(INPUT_DIR, VIDEO_EXTENSIONS, "video")
        background = Path(background_setting) if background_setting else choose_file(INPUT_DIR, IMAGE_EXTENSIONS, "background image")
        output_name = OUTPUT_NAME.strip() or f"{source.stem}_NEW_BACKGROUND.mp4"
        preview_value = os.environ.get("BACKGROUND_PREVIEW_SECONDS", "").strip()
        preview_seconds = float(preview_value) if preview_value else None
        analysis_fps = PROCESSING_FPS
        if preview_seconds:
            output_name = f"{source.stem}_NEW_BACKGROUND_PREVIEW_{int(preview_seconds)}S.mp4"
        output = OUTPUT_DIR / output_name

        if not source.exists():
            raise FileNotFoundError(f"Video not found: {source}")
        if not background.exists():
            raise FileNotFoundError(f"Background image not found: {background}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        ffmpeg = ffmpeg_path()
        ffprobe = ffprobe_path()

        log_print(f"Video:      {source.name}")
        log_print(f"Background: {background.name}")
        log_print(f"Output:     {output.name}")

        # ----------------------------------------------------------------
        # Use a PERSISTENT temp folder (no TemporaryDirectory auto-delete).
        # This means if the script crashes, the extracted frames survive and
        # you can resume. Old temp folders are cleaned up at the end only
        # after a successful run.
        # ----------------------------------------------------------------
        video_stem_safe = "".join(c if c.isalnum() else "_" for c in source.stem)[:30]
        if MODEL_NAME == "rvm_mobilenetv3":
            video_stem_safe += "_rvm"
        timestamp_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Estimate expected frame count so resume only picks the right folder.
        # preview_seconds is exact; for a full run we probe the video duration.
        if preview_seconds:
            expected_frames = int(preview_seconds * analysis_fps)
        else:
            try:
                probe = subprocess.run(
                    [ffprobe, "-v", "error",
                     "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
                    capture_output=True, text=True, check=True,
                )
                expected_frames = int(float(probe.stdout.strip()) * analysis_fps)
            except Exception:
                expected_frames = None  # unknown — match any folder

        # Try to find an existing temp folder to resume from
        existing_temp = find_existing_temp(video_stem_safe, expected_frames)
        if existing_temp:
            temp = existing_temp
            frames = temp / "frames"
            keyed = temp / "keyed"
            log_print(f"Resuming from: {temp}")
        else:
            temp = TEMP_DIR / f"bg_{video_stem_safe}_{timestamp_tag}"
            frames = temp / "frames"
            keyed = temp / "keyed"
            temp.mkdir(parents=True, exist_ok=True)
            frames.mkdir()
            keyed.mkdir()
            log_print(f"Temp folder: {temp}")

        # ----------------------------------------------------------------
        # Step 1: Extract frames (skip if already done)
        # ----------------------------------------------------------------
        existing_frames = sorted(frames.glob("*.png"))
        if existing_frames:
            log_print(f"Frames already extracted ({len(existing_frames)} found), skipping extraction.")
        else:
            log_print("Extracting frames...")
            extract_command = [
                ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
            ]
            if preview_seconds:
                extract_command += ["-t", str(preview_seconds)]
            extract_command += [
                "-i", str(source),
                "-vf", (
                    f"scale={int(SEGMENTATION_SIZE[0] * SOURCE_PAD_SCALE)}:{int(SEGMENTATION_SIZE[1] * SOURCE_PAD_SCALE)}:"
                    "force_original_aspect_ratio=decrease,"
                    f"pad={SEGMENTATION_SIZE[0]}:{SEGMENTATION_SIZE[1]}:(ow-iw)/2:(oh-ih)/2:color=black"
                ),
                "-r", str(analysis_fps), str(frames / "%06d.png"),
            ]
            run(extract_command)
            existing_frames = sorted(frames.glob("*.png"))
            log_print(f"Extracted {len(existing_frames)} frames.")

        frame_paths = sorted(frames.glob("*.png"))
        if not frame_paths:
            raise RuntimeError(f"No frames were extracted! Check your input video: {source}")

        # ----------------------------------------------------------------
        # Step 2: Background removal (skips already-keyed frames)
        # ----------------------------------------------------------------
        already_keyed = len(list(keyed.glob("*.png")))
        tasks = [(str(frame), str(keyed / frame.name)) for frame in frame_paths]
        remaining = [(s, t) for s, t in tasks if not Path(t).exists()]
        log_print(f"Removing backgrounds: {len(tasks)} total, {already_keyed} already done, {len(remaining)} remaining...")

        if remaining:
            if MODEL_NAME == "rvm_mobilenetv3":
                # RVM's recurrent state must remain in one ordered process;
                # parallel workers would destroy the temporal memory benefit.
                process_rvm_sequence(frame_paths, keyed)
            else:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=WORKERS,
                    initializer=init_worker,
                    initargs=(MODEL_NAME,),
                ) as pool:
                    for index, _ in enumerate(pool.map(remove_background, tasks), 1):
                        if index % 30 == 0 or index == len(tasks):
                            log_print(f"Processed {index}/{len(tasks)} frames", flush=True)
        else:
            log_print("All frames already keyed, skipping background removal.")

        # Verify all frames were keyed
        keyed_paths = sorted(keyed.glob("*.png"))
        if len(keyed_paths) != len(frame_paths):
            raise RuntimeError(
                f"Background removal incomplete! "
                f"Expected {len(frame_paths)} keyed frames but only found {len(keyed_paths)}. "
                f"Temp folder preserved at: {temp}"
            )

        # ----------------------------------------------------------------
        # Step 3: Edge cleanup
        # ----------------------------------------------------------------
        log_print("Stabilizing and cleaning subject edges...")
        clean_matte_sequence(keyed_paths)

        # ----------------------------------------------------------------
        # Step 4: Composite
        # ----------------------------------------------------------------
        bg_path = temp / "background.png"
        bg = ImageOps.fit(
            Image.open(background).convert("RGB"), (1920, 1080),
            method=Image.Resampling.LANCZOS, centering=(0.5, 0.5),
        )
        bg.save(bg_path)

        display_scale = SUBJECT_SCALE
        subject_width = int(1920 * display_scale)
        subject_height = int(1080 * display_scale)
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

        log_print("Compositing the presenter over the new background...")
        composite_command = [
            ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
            "-framerate", str(analysis_fps), "-i", str(keyed / "%06d.png"),
            "-loop", "1", "-i", str(bg_path),
            "-i", str(source),
            "-filter_complex", filter_graph,
            "-map", "[v]", "-map", "2:a?",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-ar", "48000", "-ac", "2", "-r", "30",
            "-shortest", "-movflags", "+faststart",
        ]
        if preview_seconds:
            composite_command += ["-t", str(preview_seconds)]
        composite_command += [str(output)]
        run(composite_command)

        # ----------------------------------------------------------------
        # Step 5: Verify output and clean up temp
        # ----------------------------------------------------------------
        if not output.exists() or output.stat().st_size < 1024:
            raise RuntimeError(
                f"Output file was not created or is too small: {output}\n"
                f"Temp folder preserved at: {temp}"
            )

        output_mb = output.stat().st_size / (1024 * 1024)
        log_print(f"\n✅ SUCCESS! Output saved to:")
        log_print(f"   {output.resolve()}")
        log_print(f"   Size: {output_mb:.1f} MB")

        # Clean up the temp folder only after confirmed success
        try:
            shutil.rmtree(temp)
            log_print(f"Temp folder cleaned up.")
        except Exception as e:
            log_print(f"Warning: could not clean temp folder: {e}")

        log.close()
        return 0

    except Exception as exc:
        log_print(f"\n❌ ERROR: {exc}")
        log_print(traceback.format_exc())
        log_print(f"\nFull log saved to: {LOG_FILE}")
        log.close()
        input("\nPress Enter to close...")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
