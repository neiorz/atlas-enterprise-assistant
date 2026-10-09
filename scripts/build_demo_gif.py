"""Assemble docs/demo.gif from ordered screenshots (FR: 60-90s walkthrough).

Why this exists: the brief requires a `docs/demo.gif` showing a walkthrough, and
a hand-recorded screen capture cannot be reproduced by a teammate or re-run
after a UI change. This builds the GIF deterministically from the PNGs in
`docs/screenshots/`, so the demo is regenerated with the same command that
regenerates the architecture poster:

    python scripts/build_demo_gif.py

Every frame is a real screenshot of the running Chainlit app; the only thing
this script adds is the caption bar and the timing. Per-frame durations are
supplied by `captions.json` so the total lands inside the 60-90s window
without editing this file.

Usage:
    python scripts/build_demo_gif.py [--seconds-per-frame N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SHOTS_DIR = ROOT / "docs" / "screenshots"
CAPTIONS_PATH = SHOTS_DIR / "captions.json"
DEFAULT_OUT = ROOT / "docs" / "demo.gif"

# Brand matches scripts/build_poster.py so the GIF and poster read as one set.
BRAND = (0, 71, 255)
BAR_BG = (13, 16, 23)
TEXT = (240, 242, 247)
MUTED = (150, 157, 172)
CAPTION_BAR_PX = 96
FRAME_WIDTH = 1400
TOTAL_BUDGET_S = (60, 90)  # brief's required walkthrough length


def load_font(size: int) -> ImageFont.ImageFont:
    """Best available font; falls back to Pillow's bitmap default."""
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_caption_bar(width: int, title: str, detail: str) -> Image.Image:
    """Render the caption strip shown under each screenshot."""
    bar = Image.new("RGB", (width, CAPTION_BAR_PX), BAR_BG)
    draw = ImageDraw.Draw(bar)
    draw.rectangle([0, 0, 8, CAPTION_BAR_PX], fill=BRAND)

    title_font = load_font(30)
    detail_font = load_font(22)
    draw.text((34, 14), title, font=title_font, fill=TEXT)
    if detail:
        draw.text((34, 54), detail, font=detail_font, fill=MUTED)
    return bar


def fit(image: Image.Image, width: int) -> Image.Image:
    """Scale to `width` keeping aspect, so mixed screenshot sizes align."""
    if image.width == width:
        return image.convert("RGB")
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.LANCZOS).convert("RGB")


def build_frames() -> list[tuple[Image.Image, float]]:
    """Compose (frame, duration_seconds) for every ordered screenshot."""
    captions = json.loads(CAPTIONS_PATH.read_text(encoding="utf-8"))
    shots = sorted(SHOTS_DIR.glob("*.png"))
    if not shots:
        sys.exit(f"No screenshots in {SHOTS_DIR} — run the app and capture some first.")

    frames: list[tuple[Image.Image, float]] = []
    for shot in shots:
        meta = captions.get(shot.stem, {})
        title = meta.get("title", shot.stem)
        detail = meta.get("detail", "")
        seconds = float(meta.get("seconds", 8))

        base = fit(Image.open(shot), FRAME_WIDTH)
        bar = draw_caption_bar(FRAME_WIDTH, title, detail)
        canvas = Image.new("RGB", (FRAME_WIDTH, base.height + CAPTION_BAR_PX), BAR_BG)
        canvas.paste(base, (0, 0))
        canvas.paste(bar, (0, base.height))
        frames.append((canvas, seconds))

    total = sum(s for _, s in frames)
    lo, hi = TOTAL_BUDGET_S
    if not lo <= total <= hi:
        print(
            f"WARNING: total is {total:.0f}s, brief wants {lo}-{hi}s. "
            f"Adjust `seconds` in {CAPTIONS_PATH.relative_to(ROOT)}.",
            file=sys.stderr,
        )
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output path (default {DEFAULT_OUT.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--seconds-per-frame",
        type=float,
        default=None,
        help="override every caption's duration with one value",
    )
    args = parser.parse_args()

    if not CAPTIONS_PATH.exists():
        sys.exit(f"Missing {CAPTIONS_PATH}")

    frames = build_frames()
    if args.seconds_per_frame:
        frames = [(img, args.seconds_per_frame) for img, _ in frames]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # loop=0 => infinite loop; disposal=2 => full-frame redraw (no ghosting
    # when a frame is shorter than the previous one).
    # frames holds (image, duration) tuples, so index the image out of frame 0.
    frames[0][0].save(
        args.out,
        save_all=True,
        append_images=[f for f, _ in frames[1:]],
        duration=[round(s * 1000) for _, s in frames],
        loop=0,
        disposal=2,
        optimize=True,
    )

    total = sum(s for _, s in frames)
    size_kb = args.out.stat().st_size // 1024
    try:
        shown = args.out.relative_to(ROOT)
    except ValueError:
        shown = args.out  # --out was pointed outside the repo
    print(f"{shown}: {len(frames)} frames, {total:.0f}s, {size_kb} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
