# Background Replacer

Turn a presenter video into a polished, composited scene with a new background—directly from Windows.

This project combines human-focused segmentation, edge cleanup, contact-shadow treatment, tone matching, subtle shared grain, background fitting, and audio preservation into one repeatable workflow.

## What it does

- Removes the original video background with a human-segmentation mask.
- Cleans hair, shoulders, clothing, ears, face, and moving-hand edges.
- Composites the subject over a supplied image with controlled blur.
- Fits the background to 1920×1080 without stretching.
- Preserves the source video's audio when available.
- Keeps large temporary frame caches on the configured D: drive.
- Includes a fast 30-second preview workflow.

## Quick start

1. Put one presenter video and one background image in [`input/`](input/).
2. Double-click [`RUN_BACKGROUND_REPLACER.bat`](RUN_BACKGROUND_REPLACER.bat).
3. If multiple videos or images are present, choose them by number when prompted.
4. Find the rendered result in the generated output location.

For a quick test, run [`RUN_30_SECOND_PREVIEW.bat`](RUN_30_SECOND_PREVIEW.bat).

## Requirements

- Windows
- Python 3.11 or a compatible Python 3 installation
- FFmpeg available to the workflow
- Python packages used by `replace_background.py`
- Enough disk space for temporary extracted frames

The first run may download the `rembg` human-segmentation model automatically.

## Advanced usage

Run with explicit paths from PowerShell:

```powershell
$env:BACKGROUND_VIDEO_PATH="D:\\path\\to\\presenter.mp4"
$env:BACKGROUND_IMAGE_PATH="D:\\path\\to\\background.png"
python .\replace_background.py
```

The main configuration values—processing FPS, segmentation size, subject scale, and worker count—are documented in [`replace_background.py`](replace_background.py). Higher quality settings increase render time and memory use.

## Project layout

```text
background_replacer/
├── replace_background.py       # Main rendering pipeline
├── resume_saved_render.py      # Resume support for saved renders
├── diagnose.py                 # Diagnostics helper
├── RUN_BACKGROUND_REPLACER.bat # Full render launcher
├── RUN_30_SECOND_PREVIEW.bat   # Preview launcher
├── input/                      # Source video and background image
└── vendor/                     # Included RobustVideoMatting source
```

Generated caches, Python bytecode, logs, and rendered videos are intentionally ignored by Git so the public repository stays clean and reproducible.

## License

This repository contains vendored RobustVideoMatting source under `vendor/RobustVideoMatting/`. Review its included license before redistribution. The project code in this repository has not been assigned a separate license yet.
