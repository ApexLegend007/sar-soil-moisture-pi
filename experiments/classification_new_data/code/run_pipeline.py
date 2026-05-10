#!/usr/bin/env python3
"""
run_pipeline.py — Interactive experiment runner with live progress tracking.

Usage:
    # Run full pipeline
    /snap/bin/astral-uv.uv run python experiments/classification_new_data/code/run_pipeline.py

    # Resume from a specific notebook
    /snap/bin/astral-uv.uv run python experiments/classification_new_data/code/run_pipeline.py --from ann_censored

    # Skip specific notebooks
    /snap/bin/astral-uv.uv run python experiments/classification_new_data/code/run_pipeline.py --skip exploration_eos exploration_sentinel

    # Run export to Excel only (after experiments are done)
    /snap/bin/astral-uv.uv run python experiments/classification_new_data/code/run_pipeline.py --export-only
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                            SpinnerColumn, TaskProgressColumn,
                            TimeElapsedColumn, TextColumn)
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data"
OUTPUT_DIR = CODE_DIR.parent / "output"

NOTEBOOKS = [
    ("exploration_eos",                            [DATA_DIR / "EOS-04_datasheet.xlsx"]),
    ("exploration_sentinel",                       [DATA_DIR / "sentinel-1.xlsx"]),
    ("classification_censored",                    []),
    ("classification_uncensored",                  []),
    ("classical_ml_censored",                      []),
    ("classical_ml_uncensored",                    []),
    ("ann_censored",                               []),
    ("ann_uncensored",                             []),
    ("pi_estimation_censored",                     []),
    ("pi_estimation_uncensored",                   []),
    ("conformal_regression_censored",              []),
    ("conformal_regression_uncensored",            []),
    ("conformalized_quantile_regression_uncensored", []),
    ("quantile_regression_tau_tuning_uncensored",  []),
    ("quantile_svr_HP_tuning",                     []),
]

# Rough prior estimates (seconds) — updated dynamically as notebooks complete
PRIOR_ESTIMATES = {
    "exploration_eos":                              90,
    "exploration_sentinel":                         90,
    "classification_censored":                     120,
    "classification_uncensored":                   120,
    "classical_ml_censored":                       180,
    "classical_ml_uncensored":                     180,
    "ann_censored":                                180,
    "ann_uncensored":                              180,
    "pi_estimation_censored":                      240,
    "pi_estimation_uncensored":                    240,
    "conformal_regression_censored":               150,
    "conformal_regression_uncensored":             150,
    "conformalized_quantile_regression_uncensored": 200,
    "quantile_regression_tau_tuning_uncensored":   200,
    "quantile_svr_HP_tuning":                     3600,
}

STATUS_STYLE = {
    "pending":  ("dim white",    "○"),
    "skipped":  ("yellow",       "⏭"),
    "running":  ("bold cyan",    "◉"),
    "passed":   ("bold green",   "✓"),
    "failed":   ("bold red",     "✗"),
}

console = Console()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class NotebookState:
    def __init__(self, name: str, prereqs: list):
        self.name      = name
        self.prereqs   = prereqs
        self.status    = "pending"
        self.elapsed   = 0.0
        self.error_tail = ""

    @property
    def display_name(self):
        return self.name.replace("_", " ")


# ---------------------------------------------------------------------------
# ETA calculation
# ---------------------------------------------------------------------------

def compute_eta(states: list[NotebookState], estimates: dict[str, float]) -> float:
    """Return estimated seconds remaining for all non-completed notebooks."""
    completed = [(s.name, s.elapsed) for s in states if s.status == "passed"]

    # Build scale factor: actual vs prior for completed notebooks
    scale = 1.0
    if completed:
        ratios = []
        for name, elapsed in completed:
            prior = PRIOR_ESTIMATES.get(name, 180)
            if prior > 0:
                ratios.append(elapsed / prior)
        if ratios:
            scale = sum(ratios) / len(ratios)

    remaining = sum(
        estimates.get(s.name, 180) * scale
        for s in states
        if s.status in ("pending", "running")
    )
    return remaining


# ---------------------------------------------------------------------------
# Rich renderable
# ---------------------------------------------------------------------------

def build_display(states: list[NotebookState], overall_progress: Progress,
                  current_start: Optional[float]) -> Table:
    now = time.time()

    # ── Notebook table ──────────────────────────────────────────────────────
    table = Table(box=box.ROUNDED, expand=True, show_header=True,
                  header_style="bold white on #1a1a2e", border_style="bright_black")
    table.add_column("#",        width=3,  justify="right", style="dim")
    table.add_column("Notebook", min_width=48)
    table.add_column("Status",   width=10, justify="center")
    table.add_column("Duration", width=10, justify="right")

    for i, s in enumerate(states, 1):
        style_color, icon = STATUS_STYLE[s.status]

        if s.status == "running" and current_start:
            elapsed = now - current_start
            dur_str = _fmt_time(elapsed) + "…"
            style_color = "bold cyan"
        elif s.status in ("passed", "failed"):
            dur_str = _fmt_time(s.elapsed)
        else:
            dur_str = "—"

        status_text = Text(f"{icon} {s.status.upper()}", style=style_color)
        table.add_row(str(i), s.display_name, status_text, dur_str)

    # ── ETA panel ───────────────────────────────────────────────────────────
    estimates = {s.name: PRIOR_ESTIMATES.get(s.name, 180) for s in states}
    eta_secs  = compute_eta(states, estimates)
    done      = sum(1 for s in states if s.status in ("passed", "skipped", "failed"))
    total     = len(states)

    eta_str   = _fmt_time(eta_secs) if eta_secs > 0 else "—"
    passed    = sum(1 for s in states if s.status == "passed")
    failed    = sum(1 for s in states if s.status == "failed")
    skipped   = sum(1 for s in states if s.status == "skipped")

    summary = (
        f"  [green]✓ {passed} passed[/]  "
        f"[red]✗ {failed} failed[/]  "
        f"[yellow]⏭ {skipped} skipped[/]  "
        f"[dim]○ {total - done} pending[/]   "
        f"[bold white]ETA: {eta_str}[/]"
    )

    layout = Table.grid(expand=True)
    layout.add_column()
    layout.add_row(overall_progress)
    layout.add_row(Text.from_markup(summary))
    layout.add_row(Text(""))
    layout.add_row(table)

    return Panel(layout, title="[bold white] SAR Soil Moisture — Experiment Pipeline [/]",
                 border_style="bright_blue", padding=(0, 1))


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_notebook(state: NotebookState, env: dict) -> bool:
    nb_path = CODE_DIR / f"{state.name}.ipynb"
    proc = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.timeout=7200",
            "--ExecutePreprocessor.kernel_name=python3",
            str(nb_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(CODE_DIR),
        env=env,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:].strip()
        state.error_tail = tail
    return proc.returncode == 0


def _build_ld_library_path() -> str:
    """Prepend venv NVIDIA cuDNN/cuBLAS wheel paths so TF 2.21 finds cuDNN 9.3."""
    import site
    venv_sp = Path(site.getsitepackages()[0])
    nvidia_lib_dirs = [
        str(venv_sp / "nvidia" / "cudnn" / "lib"),
        str(venv_sp / "nvidia" / "cublas" / "lib"),
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = nvidia_lib_dirs + ["/usr/lib/x86_64-linux-gnu"] + ([existing] if existing else [])
    return ":".join(p for p in parts if p)


def run_pipeline(skip_names: list[str], start_from: Optional[str], export_after: bool):
    env = {**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path()}

    states = [NotebookState(name, prereqs) for name, prereqs in NOTEBOOKS]

    # Mark skips / before start_from
    reached_start = (start_from is None)
    for s in states:
        if not reached_start:
            if s.name == start_from:
                reached_start = True
            else:
                s.status = "skipped"
                s.elapsed = 0
        if s.name in skip_names:
            s.status = "skipped"

    # Overall progress bar
    overall = Progress(
        SpinnerColumn(),
        TextColumn("[bold white]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        expand=False,
    )
    task_id = overall.add_task("Running notebooks", total=len(states))
    done_count = sum(1 for s in states if s.status == "skipped")
    overall.update(task_id, completed=done_count)

    current_start: Optional[float] = None
    pipeline_start = time.time()

    with Live(build_display(states, overall, None), refresh_per_second=4, console=console) as live:
        for s in states:
            if s.status == "skipped":
                overall.update(task_id, advance=1,
                               description=f"Skipped {s.display_name}")
                live.update(build_display(states, overall, None))
                continue

            # Check prereqs
            missing = [str(p) for p in s.prereqs if not Path(p).exists()]
            if missing:
                s.status = "skipped"
                s.elapsed = 0
                overall.update(task_id, advance=1,
                               description=f"Skipped {s.display_name} (missing files)")
                live.update(build_display(states, overall, None))
                continue

            # Mark running
            s.status = "running"
            current_start = time.time()
            overall.update(task_id, description=f"Running  {s.display_name}")
            live.update(build_display(states, overall, current_start))

            ok = run_notebook(s, env)

            s.elapsed = time.time() - current_start
            s.status = "passed" if ok else "failed"
            current_start = None

            overall.update(task_id, advance=1,
                           description=f"{'Done' if ok else 'FAILED'} {s.display_name}")
            live.update(build_display(states, overall, None))

            if not ok:
                live.stop()
                console.print(f"\n[bold red]✗ FAILED:[/] {s.name}")
                console.print(f"[dim]{s.error_tail}[/]")
                import sys as _sys
                if _sys.stdin.isatty():
                    try:
                        console.print("\n[yellow]Continue with next notebook? [Y/n][/] ", end="")
                        ans = input().strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        ans = "y"
                else:
                    ans = "y"
                    console.print("\n[yellow]Non-interactive mode — continuing automatically.[/]")
                if ans == "n":
                    break
                live.start()

    # ── Summary ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    passed  = [s for s in states if s.status == "passed"]
    failed  = [s for s in states if s.status == "failed"]
    skipped = [s for s in states if s.status == "skipped"]

    console.print()
    console.rule("[bold white]Pipeline Complete")
    console.print(f"  Total time : [bold]{_fmt_time(total_elapsed)}[/]")
    console.print(f"  [green]Passed [/]: {len(passed)}")
    console.print(f"  [red]Failed [/]: {len(failed)}")
    console.print(f"  [yellow]Skipped[/]: {len(skipped)}")

    if failed:
        console.print("\n[bold red]Failed notebooks:[/]")
        for s in failed:
            console.print(f"  • {s.name}")

    # Per-notebook timing table
    console.print()
    t = Table("Notebook", "Status", "Duration", box=box.SIMPLE_HEAVY)
    for s in states:
        color = {"passed": "green", "failed": "red", "skipped": "yellow"}.get(s.status, "white")
        icon, _ = STATUS_STYLE[s.status][1], ""
        t.add_row(s.name, f"[{color}]{s.status.upper()}[/]",
                  _fmt_time(s.elapsed) if s.elapsed else "—")
    console.print(t)

    # ── Excel export ─────────────────────────────────────────────────────────
    if export_after or (not failed and _ask_export()):
        _run_export(env)


def _ask_export() -> bool:
    console.print("\n[bold white]Export results to Excel workbooks? [Y/n][/] ", end="")
    try:
        return input().strip().lower() != "n"
    except (EOFError, KeyboardInterrupt):
        return False


def _run_export(env: dict):
    console.print("\n[bold cyan]Exporting metrics + plots to Excel…[/]")
    export_script = CODE_DIR / "export_to_excel.py"
    start = time.time()
    proc = subprocess.run(
        [sys.executable, str(export_script)],
        cwd=str(CODE_DIR),
        env=env,
    )
    elapsed = time.time() - start
    if proc.returncode == 0:
        console.print(f"[green]✓ Export complete[/] ({_fmt_time(elapsed)})")
        console.print(f"  Workbooks saved to: [bold]{OUTPUT_DIR}/[/]")
    else:
        console.print("[red]✗ Export failed[/]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive SAR soil moisture experiment pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--from",   dest="start_from", metavar="NOTEBOOK",
                        help="Resume pipeline from this notebook (skip all before it)")
    parser.add_argument("--skip",   dest="skip", nargs="+", metavar="NOTEBOOK", default=[],
                        help="Skip specific notebooks by name")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip all notebooks and only run Excel export")
    args = parser.parse_args()

    if args.export_only:
        env = {**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path()}
        _run_export(env)
        return

    all_names = [n for n, _ in NOTEBOOKS]
    if args.start_from and args.start_from not in all_names:
        console.print(f"[red]Unknown notebook:[/] {args.start_from}")
        console.print(f"Valid names: {', '.join(all_names)}")
        sys.exit(1)

    for name in args.skip:
        if name not in all_names:
            console.print(f"[red]Unknown notebook to skip:[/] {name}")
            sys.exit(1)

    try:
        run_pipeline(
            skip_names=args.skip,
            start_from=args.start_from,
            export_after=False,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/]")


if __name__ == "__main__":
    main()
