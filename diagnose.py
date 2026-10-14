"""Diagnostic: find exactly where clean_matte_sequence crashes."""
import sys
import traceback
from pathlib import Path
from PIL import Image, ImageChops, ImageFilter

KEYED = Path(r"D:\missionasset\background_replacer\temp\bg_MVI_1kjednkckj352___Trim_20260902_231239\keyed")

EDGE_EROSION_SIZE = 1
EDGE_FEATHER_RADIUS = 0.2
TEMPORAL_ALPHA_BLEND = 0.0
SHADOW_OPACITY = 46
SHADOW_BLUR_RADIUS = 18
SHADOW_Y_OFFSET = 10
LIGHT_WRAP_OPACITY = 0
LIGHT_WRAP_COLOR = (218, 158, 88)

paths = sorted(KEYED.glob("*.png"))
print(f"Total keyed frames: {len(paths)}", flush=True)

# Check all frames for corruption
bad = []
for p in paths:
    try:
        with Image.open(p) as img:
            img.verify()
    except Exception as e:
        bad.append((p, str(e)))

if bad:
    print(f"CORRUPT FRAMES FOUND: {len(bad)}", flush=True)
    for p, err in bad[:10]:
        print(f"  {p.name}: {err}", flush=True)
else:
    print("All frames verified OK.", flush=True)

# Now try the full clean_matte_sequence on frame 1
print("Testing clean_matte on frame 1...", flush=True)
try:
    path = paths[0]
    cutout = Image.open(path).convert("RGBA")
    print(f"  size={cutout.size} mode={cutout.mode}", flush=True)
    alpha = cutout.getchannel("A")
    alpha = alpha.filter(ImageFilter.MinFilter(EDGE_EROSION_SIZE))
    alpha = alpha.filter(ImageFilter.GaussianBlur(EDGE_FEATHER_RADIUS))
    alpha = alpha.point(lambda v: 0 if v < 32 else v)
    cutout.putalpha(alpha)

    shadow_blur = alpha.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_RADIUS))
    shifted_shadow = Image.new("L", alpha.size, 0)
    shifted_shadow.paste(shadow_blur, (0, SHADOW_Y_OFFSET))
    shadow_alpha = shifted_shadow.point(lambda v: v * SHADOW_OPACITY // 255)
    shadow = Image.new("RGBA", cutout.size, (7, 5, 4, 0))
    shadow.putalpha(shadow_alpha)

    wrap_source = alpha.filter(ImageFilter.GaussianBlur(5))
    wrap_alpha = ImageChops.subtract(wrap_source, alpha)
    wrap_alpha = wrap_alpha.point(lambda v: v * LIGHT_WRAP_OPACITY // 255)
    wrap = Image.new("RGBA", cutout.size, (*LIGHT_WRAP_COLOR, 0))
    wrap.putalpha(wrap_alpha)

    layer = Image.new("RGBA", cutout.size, (0, 0, 0, 0))
    layer.alpha_composite(shadow)
    layer.alpha_composite(wrap)
    layer.alpha_composite(cutout)
    print("  clean_matte frame 1: SUCCESS", flush=True)
except Exception as e:
    print(f"  clean_matte FAILED: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("Diagnostic complete - no issues found!", flush=True)
