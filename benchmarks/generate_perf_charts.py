#!/usr/bin/env python3
"""Run API benchmarks and write the homepage representative chart for docs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
OUT_DIR = ROOT / "docs" / "generated" / "performance"

# Imaginary brand palette (see docs/stylesheets/imaginary.css)
COLORS = {
    "numpy": "#777067",  # ink-400
    "grumpy": "#4a6b52",  # green-light
    "awkward": "#484240",  # ink-600
    "paper": "#faf9f7",
    "plot": "#f5f3ef",
    "grid": "#ddd8ce",
    "text": "#1e1b19",
}

REPRESENTATIVE_OPS = [
    "(a * b).sum()",
    "a.sum()",
    "isin",
    "fancy get + sum",
]

LIB_SERIES = (
    ("Grumpy", "grumpy_ms", "grumpy"),
    ("NumPy", "numpy_ms", "numpy"),
    ("Awkward", "awkward_ms", "awkward"),
)


def _ensure_imports() -> None:
    try:
        import grumpy  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "grumpy is not importable. Build the extension first:\n"
            "  maturin develop --release"
        ) from exc
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "plotly is required for docs charts:\n"
            "  pip install plotly"
        ) from exc


def _run_ragged_api_benchmark(*, nrows: int, ncols: int, nfancy: int, warmup: int, repeats: int, seed: int) -> Path:
    sys.path.insert(0, str(BENCH_DIR))
    from benchmark_ragged_api import main as run_api  # noqa: WPS433

    json_path = OUT_DIR / "ragged_api.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    argv = [
        "--nrows",
        str(nrows),
        "--ncols",
        str(ncols),
        "--nfancy",
        str(nfancy),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
        "--json",
        str(json_path),
    ]
    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):
        code = run_api(argv)
    if code != 0:
        raise SystemExit(f"benchmark_ragged_api.py failed with exit code {code}")
    return json_path


def _load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _fig_layout(title: str, *, height: int = 420) -> dict[str, Any]:
    return dict(
        title=dict(text=title, font=dict(family="IBM Plex Sans, sans-serif", size=16, color=COLORS["text"])),
        font=dict(family="IBM Plex Sans, sans-serif", color=COLORS["text"]),
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["plot"],
        height=height,
        margin=dict(l=48, r=24, t=56, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"], tickfont=dict(size=11)),
        yaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"], tickfont=dict(size=11), title="Time (ms)"),
        barmode="group",
    )


def _bar_chart_ms(cases: list[dict[str, Any]], title: str, *, height: int = 420):
    import plotly.graph_objects as go

    names = [c["name"] for c in cases]
    fig = go.Figure()
    for lib, key, color_key in LIB_SERIES:
        fig.add_bar(
            name=lib,
            x=names,
            y=[c[key] for c in cases],
            marker_color=COLORS[color_key],
        )
    fig.update_layout(**_fig_layout(title, height=height))
    fig.update_xaxes(tickangle=-25)
    return fig


def _summary_chart(cases: list[dict[str, Any]]):
    selected = [c for c in cases if c["name"] in REPRESENTATIVE_OPS]
    if not selected:
        selected = cases[:4]
    return _bar_chart_ms(selected, "Representative ops — public API (ms)", height=440)


def _write_html(fig, path: Path, *, include_plotlyjs: bool | str = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        fig.to_html(full_html=True, include_plotlyjs=include_plotlyjs),
        encoding="utf-8",
    )


def _cleanup_stale_artifacts() -> None:
    keep = {"summary.html", "ragged_api.json", "manifest.json"}
    if not OUT_DIR.is_dir():
        return
    for path in OUT_DIR.iterdir():
        if path.name not in keep:
            path.unlink()


def generate_charts(report: dict[str, Any]) -> None:
    cases = report["cases"]
    _cleanup_stale_artifacts()
    _write_html(_summary_chart(cases), OUT_DIR / "summary.html", include_plotlyjs="cdn")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": report.get("suite"),
        "nrows": report.get("nrows"),
        "ncols": report.get("ncols"),
        "n_elements": report.get("n_elements"),
        "python": report.get("python"),
        "numpy": report.get("numpy"),
        "awkward": report.get("awkward"),
        "case_count": len(cases),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate homepage benchmark chart from API benchmarks.")
    ap.add_argument("--nrows", type=int, default=int(os.environ.get("GRUMPY_BENCH_NROWS", "4096")))
    ap.add_argument("--ncols", type=int, default=int(os.environ.get("GRUMPY_BENCH_NCOLS", "256")))
    ap.add_argument("--nfancy", type=int, default=int(os.environ.get("GRUMPY_BENCH_NFANCY", "4096")))
    ap.add_argument("--warmup", type=int, default=int(os.environ.get("GRUMPY_BENCH_WARMUP", "3")))
    ap.add_argument("--repeats", type=int, default=int(os.environ.get("GRUMPY_BENCH_REPEATS", "7")))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="Regenerate charts from existing docs/generated/performance/ragged_api.json",
    )
    args = ap.parse_args(argv)

    _ensure_imports()

    json_path = OUT_DIR / "ragged_api.json"
    if not args.skip_run:
        json_path = _run_ragged_api_benchmark(
            nrows=args.nrows,
            ncols=args.ncols,
            nfancy=args.nfancy,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        )
    elif not json_path.is_file():
        raise SystemExit(f"--skip-run requested but {json_path} is missing")

    generate_charts(_load_report(json_path))
    print(f"Wrote homepage chart to {OUT_DIR / 'summary.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
