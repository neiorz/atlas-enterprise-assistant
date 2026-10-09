"""Measure steady-state RAM + embedding parity for fp32 vs bf16 (NFR-B4).

NFR-B4 caps the running app at 1 GB. bge-m3 in fp32 peaked ~1.8 GB, so we
check whether loading it in bfloat16 gets under the cap WITHOUT changing the
vectors enough to break retrieval quality.

Run:  python scripts/mem_check.py fp32
      python scripts/mem_check.py bf16
Then compare the printed query vectors (cosine sim should stay ~1.0).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUERY = "كم الحد الأقصى لعشاء العملاء لكل شخص؟"


def current_rss_mb() -> float:
    """Resident memory right now (not the all-time peak, which is polluted by
    the transient fp32->bf16 copy made while converting weights)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return 0.0


def rss_split_mb() -> tuple[float, float]:
    """(RssAnon, RssFile) — anonymous memory can't be reclaimed, while
    file-backed pages are the mmap'd model weights the OS can drop under
    pressure. Worth reporting separately for NFR-B4."""
    anon = file_ = 0.0
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        if line.startswith("RssAnon:"):
            anon = int(line.split()[1]) / 1024
        elif line.startswith("RssFile:"):
            file_ = int(line.split()[1]) / 1024
    return anon, file_


def main() -> int:
    precision = sys.argv[1] if len(sys.argv) > 1 else "fp32"
    import torch
    from sentence_transformers import SentenceTransformer

    baseline = current_rss_mb()
    kwargs = (
        {"model_kwargs": {"torch_dtype": torch.bfloat16}} if precision == "bf16" else {}
    )

    started = time.perf_counter()
    model = SentenceTransformer("BAAI/bge-m3", **kwargs)
    load_s = time.perf_counter() - started
    after_load = current_rss_mb()

    # encode the query plus the Arabic gold doc text so we can compare vectors
    texts = [
        QUERY,
        "الحد الأقصى لعشاء العملاء هو 2,000 جنيه مصري لكل شخص حاضر.",
        "Domestic hotel cap: 1,500 EGP per night (room + tax).",
    ]
    started = time.perf_counter()
    vecs = model.encode(texts, normalize_embeddings=True)
    enc_s = time.perf_counter() - started
    after_encode = current_rss_mb()
    anon, file_ = rss_split_mb()

    out = Path(f"/tmp/opencode/vec_{precision}.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.save(out, vecs)

    print(f"precision     : {precision}")
    print(f"model load    : {load_s:.1f}s")
    print(f"encode 3 texts: {enc_s:.2f}s")
    print(f"RSS baseline  : {baseline:.0f} MB")
    print(
        f"RSS after load: {after_load:.0f} MB  (model footprint: {after_load - baseline:.0f} MB)"
    )
    print(
        f"RSS after enc : {after_encode:.0f} MB  <-- steady-state, vs NFR-B4's 1024 MB"
    )
    print(f"  RssAnon     : {anon:.0f} MB (not reclaimable)")
    print(
        f"  RssFile     : {file_:.0f} MB (mmap'd weights, reclaimable under pressure)"
    )
    print(f"vectors saved : {out}")
    print(f"query vec[:5] : {vecs[0][:5].round(5)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
