#!/usr/bin/env python3
"""Build docs/architecture.png — the project's single "architecture + all
details" hero image used at the top of README.md.

Why this exists: graphviz lays out one connected graph globally, so a title
card, three layer diagrams and a stats footer in a single .dot file end up
scattered across ranks with edges crossing the page (tried it — the footer
rendered above the title). Instead each section is rendered on its own with
the flow direction that suits it (left-to-right bands), then stacked into one
poster with Pillow.

Rebuild after any architecture change:

    python scripts/build_poster.py
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "architecture.png"
DPI = 150
GAP = 26  # vertical gap between poster sections, px
ROW_GAP = 44  # horizontal gap between sections sharing a row, px
MARGIN = 30  # outer margin, px

BRAND = "#0047FF"  # brief's brand blue
NAVY = "#0B1B3A"
NAVY_ROW = "#132C63"
PALE = "#9DB8FF"
FONT = "DejaVu Sans"

NODE = (
    f'node [fontname="{FONT}", fontsize=11, shape=box, style="rounded,filled", '
    'color="#C7D2E8", fillcolor="white", penwidth=1.3];'
)
EDGE = (
    f'edge [fontname="{FONT}", fontsize=9, color="#667085", penwidth=1.2, '
    "arrowsize=0.7];"
)

COMMON = f"""
    bgcolor="white";
    fontname="{FONT}";
    nodesep=0.34;
    ranksep=0.50;
    {NODE}
    {EDGE}
"""


# --------------------------------------------------------------------------
# Poster sections (each rendered independently)
# --------------------------------------------------------------------------

TITLE = f"""digraph Title {{
{COMMON}
    shape=plain;
    title [shape=plain, margin=0, label=<
      <table border="1" cellborder="0" cellspacing="0" cellpadding="0" bgcolor="{NAVY}" color="{NAVY}" style="rounded">
        <tr>
          <td width="__SPACER__"> </td>
          <td align="left"><font point-size="30" color="#FFFFFF"><b>ATLAS INDUSTRIES &#8212; ENTERPRISE ASSISTANT</b></font></td>
          <td width="__SPACER__"> </td>
        </tr>
        <tr>
          <td width="__SPACER__"> </td>
          <td align="left"><font point-size="14" color="{PALE}">Agentic internal knowledge assistant &#183; bilingual English / Arabic &#183; every answer cited &#183; $0 runtime cost</font></td>
          <td width="__SPACER__"> </td>
        </tr>
        <tr>
          <td width="__SPACER__" height="10" bgcolor="{NAVY_ROW}"> </td>
          <td height="10" bgcolor="{NAVY_ROW}"> </td>
          <td width="__SPACER__" height="10" bgcolor="{NAVY_ROW}"> </td>
        </tr>
        <tr>
          <td width="__SPACER__" bgcolor="{NAVY_ROW}"> </td>
          <td align="left" bgcolor="{NAVY_ROW}">
            <table border="0" cellborder="0" cellspacing="8" cellpadding="5" align="left">
              <tr>
                <td bgcolor="{BRAND}" color="{BRAND}" style="rounded"><font point-size="12" color="#FFFFFF"><b>Python 3.12</b></font></td>
                <td bgcolor="{BRAND}" color="{BRAND}" style="rounded"><font point-size="12" color="#FFFFFF"><b>LangGraph</b></font></td>
                <td bgcolor="{BRAND}" color="{BRAND}" style="rounded"><font point-size="12" color="#FFFFFF"><b>Gemini / Groq LLM</b></font></td>
                <td bgcolor="{BRAND}" color="{BRAND}" style="rounded"><font point-size="12" color="#FFFFFF"><b>bge-m3 (1024-d)</b></font></td>
                <td bgcolor="{BRAND}" color="{BRAND}" style="rounded"><font point-size="12" color="#FFFFFF"><b>Qdrant</b></font></td>
                <td bgcolor="{BRAND}" color="{BRAND}" style="rounded"><font point-size="12" color="#FFFFFF"><b>Chainlit</b></font></td>
                <td bgcolor="{BRAND}" color="{BRAND}" style="rounded"><font point-size="12" color="#FFFFFF"><b>DeepEval</b></font></td>
                <td bgcolor="#0E9F6E" color="#0E9F6E" style="rounded"><font point-size="12" color="#FFFFFF"><b>$0 cost</b></font></td>
              </tr>
            </table>
          </td>
          <td width="__SPACER__" bgcolor="{NAVY_ROW}"> </td>
        </tr>
      </table>
    >];
}}"""

INGEST = f"""digraph Ingest {{
    rankdir=LR;
{COMMON}
    subgraph cluster_0 {{
        label=<<b>LAYER 1 &#8212; INGESTION &amp; INDEX</b>  <font point-size="10">offline &#183; run once ~70 s &#183; scripts/reset.sh</font>>;
        labeljust="l"; fontsize=14; fontcolor="{BRAND}"; color="{BRAND}";
        style="rounded"; penwidth=1.6; bgcolor="#F7F9FF"; margin=12;

        docs  [label="data/docs/{{hr, it, finance}}\\n30 documents &#183; .md .pdf .docx\\n60% en &#183; 30% ar &#183; 10% bilingual", fillcolor="#E8EEFF"];
        loads [label="loaders.py\\npypdf &#183; python-docx &#183; UTF-8\\nArabic RTL verified by hand"];
        chunk [label="chunking.py\\n800 chars / 150 overlap\\n&#8594; domain &#183; source &#183; language"];
        embed [label="embeddings.py\\nBAAI/bge-m3\\n1024-d, multilingual"];
        store [label="vectorstore.py &#183; Qdrant\\n122 vectors + policy index\\nkeyword payload filters", fillcolor="#E8EEFF"];

        docs -> loads -> chunk -> embed -> store;
        store -> hint [style=invis];
        hint [shape=plaintext, style=filled, fillcolor="white",
              label=<<font point-size="12" color="{BRAND}"><b>vector search &#8595;</b><br/><font point-size="10">feeds Layer 2</font></font>>];
    }}
}}"""

AGENT = f"""digraph Agent {{
    rankdir=TB;
{COMMON}
    nodesep=0.5; ranksep=0.30;
    subgraph cluster_0 {{
        label=<<b>LAYER 2 &#8212; LANGGRAPH AGENT</b>  <font point-size="10">src/graph.py &#183; 8 nodes &#183; typed AtlasState &#183; one run_id per turn</font>>;
        labeljust="l"; fontsize=14; fontcolor="{BRAND}"; color="{BRAND}";
        style="rounded"; penwidth=1.6; bgcolor="#FFFDF5"; margin=12;

        entry  [label=<<b>run_turn() &#8595;</b><br/><font point-size="10">from Layer 3</font>>, shape=plaintext];
        idx    [label=<<b>&#8592; vector search</b><br/><font point-size="10">from Layer 1</font>>, shape=plaintext];
        start  [label="START\\nquestion + session_id", shape=oval, fillcolor="#DFF5E7", color="#0E9F6E"];
        memload[label="memory_loader\\nreset per-turn state"];
        router [label="router\\nLLM structured output\\nhr / it / finance + confidence", fillcolor="#FFF3D6", color="#E0A100"];
        fallback[label="multi (fallback)\\nconfidence &lt; 0.55\\nfan out to all domains", fillcolor="#FDE8E8", color="#D64545"];
        retriever[label="retriever\\ndomain-filtered top-k = 5\\nsources kept for citations", fillcolor="#E8EEFF"];
        agent  [label="agent\\ntool call or answer?", fillcolor="#FFF3D6", color="#E0A100"];
        tools  [label="tools &#183; Pydantic-typed\\n&#183; policy_lookup\\n&#183; reimbursement_calculator\\n&#183; list_leave_types", fillcolor="#F3E8FF", color="#7C3AED"];
        answer [label="generate_answer\\ncited reply in user's language\\nSources: real filenames", fillcolor="#DFF5E7", color="#0E9F6E"];
        limit  [label="fallback (step limit)\\nmax 10 steps &#8594;\\nexplicit uncertainty", fillcolor="#FDE8E8", color="#D64545"];
        memwrite[label="memory_writer\\n+ close run log"];
        end    [label="END", shape=oval, fillcolor="#DFF5E7", color="#0E9F6E"];

        entry -> start [color="{BRAND}", penwidth=1.5];
        start -> memload -> router;
        router -> retriever [label="  conf &#8805; 0.55", fontcolor="{BRAND}"];
        router -> fallback  [label="  conf &lt; 0.55", style=dashed, color="#D64545", fontcolor="#D64545"];
        fallback -> retriever [style=dashed, color="#D64545"];
        idx -> retriever [color="{BRAND}", penwidth=1.5];
        retriever -> agent;
        agent -> tools  [label=" tool helps"];
        tools -> agent  [label=" result / 1 retry", style=dashed];
        agent -> answer [label=" no tool"];
        agent  -> limit [style=dashed, color="#D64545", label=" overflow", fontcolor="#D64545"];
        tools  -> limit [style=dashed, color="#D64545"];
        answer -> memwrite;
        limit  -> memwrite;
        memwrite -> end;

        {{rank=same; idx; retriever;}}
        {{rank=same; answer; limit;}}
    }}
}}"""

UI = f"""digraph UI {{
    rankdir=TB;
{COMMON}
    nodesep=0.42; ranksep=0.40;
    subgraph cluster_0 {{
        label=<<b>LAYER 3 &#8212; UI + OBSERVABILITY</b>  <font point-size="10">what the demo shows</font>>;
        labeljust="l"; fontsize=14; fontcolor="{BRAND}"; color="{BRAND}";
        style="rounded"; penwidth=1.6; bgcolor="#F5F0FF"; margin=12;

        demo    [label="scripts/smoke_demo.py &#183; 3-turn demo\\nEN policy &#8594; memory follow-up &#8594; Arabic dinner\\n1,500 / 2,000 EGP locked facts", fillcolor="#EDE4FF"];
        chainlit[label="app.py &#183; Chainlit\\nrouting / retrieval / tool step cards\\nRTL Arabic rendering &#183; New Chat control", fillcolor="#EDE4FF"];
        eval    [label="tests/evaluate.py &#183; DeepEval\\n10 gold cases &#215; 5 metrics\\n+ custom routing accuracy", fillcolor="#EDE4FF"];
        tograph [label=<<b>run_turn() &#8594;</b><br/><font point-size="10">to Layer 2</font>>, shape=plaintext];
        logs    [label="outputs/run_logs.jsonl\\none JSON event per step,\\nall tagged with run_id", shape=note, fillcolor="#FFF9DB", color="#E0A100"];
        mem     [label="session memory\\nMemorySaver\\nthread_id = session_id", fillcolor="#EDE4FF"];
        report  [label="outputs/eval_report.json", shape=note, fillcolor="#FFF9DB", color="#E0A100"];

        demo -> chainlit [style=dotted, label=" scripted run"];
        chainlit -> tograph [color="{BRAND}", penwidth=1.5];
        chainlit -> logs [style=dotted, label=" events"];
        chainlit -> mem  [style=dotted, label=" session_id"];
        eval -> report   [style=dotted];

        {{rank=same; demo;}}
        {{rank=same; chainlit; eval;}}
        {{rank=same; tograph; logs; mem; report;}}
        chainlit -> eval [style=invis, weight=20];
    }}
}}"""

REQS = f"""digraph Reqs {{
{COMMON}
    reqs [shape=plain, margin=0, label=<
      <table border="1" cellborder="0" cellspacing="0" cellpadding="0" bgcolor="#F7F9FF" color="{BRAND}" style="rounded">
        <tr>
          <td width="__SPACER__"> </td>
          <td align="left"><font point-size="15" color="{BRAND}"><b>REQUIREMENTS COVERAGE</b></font></td>
          <td width="__SPACER__"> </td>
        </tr>
        <tr>
          <td width="__SPACER__"> </td>
          <td align="left" balign="left">
            <table border="0" cellborder="0" cellspacing="0" cellpadding="5" align="left">
              <tr>
                <td align="left"><font color="#0E9F6E"><b>&#10003;</b></font></td>
                <td align="left"><font point-size="12">Citation block on every answer &#183; real filenames only</font></td>
              </tr>
              <tr>
                <td align="left"><font color="#0E9F6E"><b>&#10003;</b></font></td>
                <td align="left"><font point-size="12">English + Arabic (RTL) &#183; 3/10 gold cases are Arabic</font></td>
              </tr>
              <tr>
                <td align="left"><font color="#0E9F6E"><b>&#10003;</b></font></td>
                <td align="left"><font point-size="12">Multi-turn memory &#8212; follow-ups resolve pronouns</font></td>
              </tr>
              <tr>
                <td align="left"><font color="#0E9F6E"><b>&#10003;</b></font></td>
                <td align="left"><font point-size="12">Domain routing hr / it / finance + low-confidence fallback</font></td>
              </tr>
              <tr>
                <td align="left"><font color="#0E9F6E"><b>&#10003;</b></font></td>
                <td align="left"><font point-size="12">Deterministic tools &#183; Pydantic-typed &#183; one retry</font></td>
              </tr>
              <tr>
                <td align="left"><font color="#0E9F6E"><b>&#10003;</b></font></td>
                <td align="left"><font point-size="12">Ingestion 69 s &#8212; budget is 2 min</font></td>
              </tr>
              <tr>
                <td align="left"><font color="#0E9F6E"><b>&#10003;</b></font></td>
                <td align="left"><font point-size="12">Runtime cost $0 &#8212; free-tier LLM + local models</font></td>
              </tr>
              <tr>
                <td align="left"><font color="#0E9F6E"><b>&#10003;</b></font></td>
                <td align="left"><font point-size="12">Graceful errors &#8212; no stack traces reach the user</font></td>
              </tr>
              <tr>
                <td align="left"><font color="#E0A100"><b>&#9888;</b></font></td>
                <td align="left"><font point-size="12">RAM ~1.96 GB &#8212; bge-m3 misses the 1 GB target (documented)</font></td>
              </tr>
              <tr>
                <td align="left"><font color="#E0A100"><b>&#9888;</b></font></td>
                <td align="left"><font point-size="12">Warm turn 9.8 s &#8212; free-tier output cap, not code (documented)</font></td>
              </tr>
            </table>
          </td>
          <td width="__SPACER__"> </td>
        </tr>
      </table>
    >];
}}"""


FOOTER = f"""digraph Footer {{
{COMMON}
    shape=plain;
    footer [shape=plain, margin=0, label=<
      <table border="1" cellborder="0" cellspacing="26" cellpadding="14" bgcolor="{NAVY}" color="{NAVY}" style="rounded">
        <tr>
          <td width="__SPACER__"> </td>
          <td bgcolor="{NAVY_ROW}"><font point-size="13" color="{PALE}">TESTS</font><br/><font point-size="24" color="#FFFFFF"><b>29 passing</b></font><br/><font point-size="11" color="{PALE}">offline, no API key</font></td>
          <td bgcolor="{NAVY_ROW}"><font point-size="13" color="{PALE}">INGESTION</font><br/><font point-size="24" color="#FFFFFF"><b>69 s</b></font><br/><font point-size="11" color="{PALE}">30 docs &#8594; 122 chunks</font></td>
          <td bgcolor="{NAVY_ROW}"><font point-size="13" color="{PALE}">ARABIC RECALL</font><br/><font point-size="24" color="#FFFFFF"><b>0.701</b></font><br/><font point-size="11" color="{PALE}">FIN-009 ranked #1</font></td>
          <td bgcolor="{NAVY_ROW}"><font point-size="13" color="{PALE}">CODE HEALTH</font><br/><font point-size="24" color="#FFFFFF"><b>ruff + black</b></font><br/><font point-size="11" color="{PALE}">all src &#8804; 250 lines</font></td>
          <td bgcolor="{NAVY_ROW}"><font point-size="13" color="{PALE}">BILINGUAL</font><br/><font point-size="24" color="#FFFFFF"><b>EN + AR</b></font><br/><font point-size="11" color="{PALE}">3/10 eval cases Arabic</font></td>
          <td width="__SPACER__"> </td>
        </tr>
      </table>
    >];
}}"""


# --------------------------------------------------------------------------
# Rendering + composition
# --------------------------------------------------------------------------


def render(dot_source: str, out_path: Path) -> Image.Image:
    """Render one section to PNG via graphviz and open it."""
    dot_file = out_path.with_suffix(".dot")
    dot_file.write_text(dot_source, encoding="utf-8")
    subprocess.run(
        ["dot", "-Tpng", f"-Gdpi={DPI}", "-o", str(out_path), str(dot_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    dot_file.unlink(missing_ok=True)
    return Image.open(out_path).convert("RGB")


def render_card(dot_source: str, out_path: Path, target_w: int) -> Image.Image:
    """Render a header/footer card, stretching it to target_w.

    The card templates carry ``__SPACER__`` placeholder cells. graphviz
    honours a ``width`` on a table cell (in points), so measuring the
    natural width first and then injecting equal spacers on both sides
    makes the card full-bleed instead of a small centred box.
    """
    img = render(dot_source.replace("__SPACER__", "1"), out_path)
    if img.width >= target_w:
        return img
    # graphviz points are 1/72 inch; at -Gdpi=DPI one point is DPI/72 px.
    # Round up and iterate: graphviz rounds cell widths internally, so a
    # single pass can land a few pixels short of the target.
    for _ in range(3):
        per_side_pt = math.ceil((target_w - img.width) * 72 / DPI / 2)
        if per_side_pt <= 0:
            break
        img = render(dot_source.replace("__SPACER__", str(per_side_pt)), out_path)
        if img.width >= target_w:
            break
    return img


def stack_vertical(imgs: list[Image.Image], gap: int) -> Image.Image:
    """Vertically stack images into one, centred on a white canvas."""
    width = max(i.width for i in imgs)
    height = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for img in imgs:
        canvas.paste(img, ((width - img.width) // 2, y))
        y += img.height + gap
    return canvas


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="atlas_poster_") as tmp:
        tmpdir = Path(tmp)

        def band(name: str, src: str) -> Image.Image:
            img = render(src, tmpdir / f"{name}.png")
            print(f"  rendered {name:7s} {img.width}x{img.height}")
            return img

        layer1 = band("layer1", INGEST)
        layer2 = band("layer2", AGENT)
        layer3 = band("layer3", UI)

        # The requirements card is stretched to the UI panel's width so the
        # left column's two boxes share the same edges.
        reqs = render_card(REQS, tmpdir / "reqs.png", layer3.width)
        print(f"  rendered reqs    {reqs.width}x{reqs.height}")
        left_column = stack_vertical([layer3, reqs], GAP)

        # The diagram bands decide the poster width; the header/footer cards
        # stretch to match so every edge on the poster lines up.
        band_w = max(layer1.width, left_column.width + ROW_GAP + layer2.width)

        def card(name: str, src: str, target_w: int) -> Image.Image:
            img = render_card(src, tmpdir / f"{name}.png", target_w)
            print(f"  rendered {name:7s} {img.width}x{img.height}")
            return img

        # The UI panel sits LEFT of the agent graph so its "run_turn()" arrow
        # points right, into Layer 2, matching the real call direction.
        rows = [
            [card("title", TITLE, band_w)],
            [layer1],
            [left_column, layer2],
            [card("footer", FOOTER, band_w)],
        ]

        row_layout: list[tuple[int, list[tuple[Image.Image, int]]]] = []
        row_widths: list[int] = []
        for row in rows:
            row_widths.append(sum(i.width for i in row) + ROW_GAP * (len(row) - 1))
            x = 0
            placed: list[tuple[Image.Image, int]] = []
            for img in row:
                placed.append((img, x))
                x += img.width + ROW_GAP
            # Rows are top-aligned so neighbouring panels start on one line.
            row_layout.append((max(i.height for i in row), placed))

        canvas_w = max(row_widths) + 2 * MARGIN
        canvas_h = sum(h for h, _ in row_layout) + GAP * (len(rows) - 1) + 2 * MARGIN

        poster = Image.new("RGB", (canvas_w, canvas_h), "white")
        y = MARGIN
        for (row_h, placed), row_w in zip(row_layout, row_widths, strict=True):
            x0 = (canvas_w - row_w) // 2
            for img, x in placed:
                poster.paste(img, (x0 + x, y))
            y += row_h + GAP

        poster.save(OUT, optimize=True)
        print(f"\n  wrote {OUT.relative_to(ROOT)}  ({poster.width}x{poster.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
