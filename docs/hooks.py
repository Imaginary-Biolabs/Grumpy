"""MkDocs hooks — regenerate homepage benchmark chart before each docs build."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def on_pre_build(config, **kwargs) -> None:
    root = Path(config["docs_dir"]).resolve().parent
    script = root / "benchmarks" / "generate_perf_charts.py"
    if not script.is_file():
        return
    subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        check=True,
    )
