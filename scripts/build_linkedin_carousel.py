"""Render the LinkedIn carousel (1080x1350 portrait) from repo assets.

Why this exists: LinkedIn's feed favours portrait media, and the raw assets
are the wrong shape for it — the architecture poster is near-square and the
app screenshots are landscape 1600px wide. Posting them natively would crop
the UI badly. This letterboxes each one onto a 1080x1350 canvas with a title
and a one-line takeaway, so every slide survives the mobile feed.

Like scripts/build_poster.py and scripts/build_demo_gif.py, the slides are
declared data at the bottom of this file rather than hard-coded drawing calls,
so reordering or rewriting a caption never touches layout code.

Usage:
    python scripts/build_linkedin_carousel.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "linkedin"

# Same palette as the poster and the demo GIF, so all three read as one set.
BRAND = (0, 71, 255)
BG = (13, 16, 23)
PANEL = (23, 27, 36)
TEXT = (240, 242, 247)
MUTED = (150, 157, 172)

CANVAS = (1080, 1350)  # LinkedIn portrait
MARGIN = 72
IMAGE_GAP = 40

# (output stem, source image, slide title, one-line takeaway)
SLIDES: list[tuple[str, str, str, str]] = [
    (
        "01",
        "docs/architecture.png",
        "Atlas Industries Assistant",
        "Retrieval, routing, tools and citations — the whole graph",
    ),
    (
        "02",
        "docs/screenshots/04-tool.png",
        "It shows its work",
        "Typed arguments, the cap it checked, the file it came from",
    ),
    (
        "03",
        "docs/screenshots/03-retrieval.png",
        "One index, two languages",
        "English and Arabic policy files retrieved side by side",
    ),
    (
        "04",
        "docs/screenshots/05-cited-answer.png",
        "No source, no answer",
        "The locked fact, verbatim — with its policy file attached",
    ),
    (
        "05",
        "docs/screenshots/01-welcome.png",
        "Run it yourself",
        "Free-tier LLM · 29 tests pass with no API key required",
    ),
]


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


def fit_within(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    """Scale down to fit `box` without cropping, keeping aspect ratio."""
    bw, bh = box
    scale = min(bw / image.width, bh / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    """Greedy word wrap; Arabic filenames inside the shots need no wrapping."""
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            current = trial
        elif current:
            lines.append(current)
            current = word
        else:
            lines.append(word)  # single word wider than the canvas
            current = ""
    if current:
        lines.append(current)
    return lines


def build_slide(index: int, spec: tuple[str, str, str, str]) -> Image.Image | None:
    stem, rel_path, title, detail = spec
    source = ROOT / rel_path
    if not source.exists():
        print(f"  skip {stem}: missing {rel_path}")
        return None

    width, height = CANVAS
    canvas = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(canvas)

    # Slide counter + brand rule
    counter_font = load_font(26)
    draw.rectangle([MARGIN, MARGIN, MARGIN + 56, MARGIN + 6], fill=BRAND)
    draw.text(
        (MARGIN, MARGIN + 22),
        f"{index:02d} / {len(SLIDES):02d}",
        font=counter_font,
        fill=MUTED,
    )

    # Title, wrapped to the canvas width
    y = MARGIN + 78
    title_font = load_font(56)
    for line in wrap(draw, title, title_font, width - 2 * MARGIN):
        draw.text((MARGIN, y), line, font=title_font, fill=TEXT)
        y += 66

    # Takeaway
    y += 10
    detail_font = load_font(30)
    for line in wrap(draw, detail, detail_font, width - 2 * MARGIN):
        draw.text((MARGIN, y), line, font=detail_font, fill=MUTED)
        y += 40

    # Letterboxed screenshot, centred in the height that remains — a landscape
    # shot on a portrait canvas leaves a gap, and pinning it to the top of the
    # region reads as a layout bug rather than a deliberate crop.
    y += IMAGE_GAP
    region_h = height - y - MARGIN - 70
    shot = fit_within(Image.open(source).convert("RGB"), (width - 2 * MARGIN, region_h))
    panel = Image.new("RGB", (shot.width + 4, shot.height + 4), PANEL)
    panel.paste(shot, (2, 2))
    x = (width - panel.width) // 2
    panel_y = y + max(0, (region_h - panel.height) // 2)
    canvas.paste(panel, (x, panel_y))

    # Footer
    footer_font = load_font(24)
    draw.text(
        (MARGIN, height - MARGIN - 4),
        "github.com/neiorz/atlas-enterprise-assistant",
        font=footer_font,
        fill=MUTED,
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR, help=f"output dir (default {OUT_DIR})"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for i, spec in enumerate(SLIDES, start=1):
        slide = build_slide(i, spec)
        if slide is None:
            continue
        out = args.out_dir / f"slide-{spec[0]}.png"
        slide.save(out, optimize=True)
        written += 1
        print(f"  {out.relative_to(ROOT)}  {slide.width}x{slide.height}")

    print(f"{written} slides in {args.out_dir}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
