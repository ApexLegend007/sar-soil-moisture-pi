#!/usr/bin/env python3
"""
run_pipeline.py — Stage-wise interactive experiment runner with live progress tracking.

Stages
------
  1  Data Preparation      exploration_eos, exploration_sentinel          [CPU]
  2  Classification        classification_censored/uncensored             [CPU + XGB GPU]
  3  Classical ML          classical_ml_censored/uncensored               [GPU]
  4  ANN Regression        ann_censored/uncensored                        [GPU]
  5  PI Estimation         pi_estimation_censored/uncensored              [GPU]
  6  Conformal Regression  conformal_regression_censored/uncensored       [CPU]
  7  Advanced PI           conformalized_quantile_regression, tau_tuning  [GPU]
  8  Quantile SVR Tuning   quantile_svr_HP_tuning                         [CPU]

Usage
-----
  # Full pipeline
  /snap/bin/astral-uv.uv run python run_pipeline.py

  # Resume from a stage number
  /snap/bin/astral-uv.uv run python run_pipeline.py --from-stage 4

  # Run exactly one stage
  /snap/bin/astral-uv.uv run python run_pipeline.py --stage 4

  # Resume from a specific notebook (legacy)
  /snap/bin/astral-uv.uv run python run_pipeline.py --from ann_censored

  # Skip specific notebooks
  /snap/bin/astral-uv.uv run python run_pipeline.py --skip exploration_eos exploration_sentinel

  # Export to Excel only
  /snap/bin/astral-uv.uv run python run_pipeline.py --export-only
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                            SpinnerColumn, TaskProgressColumn,
                            TimeElapsedColumn, TextColumn)
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data"
OUTPUT_DIR = CODE_DIR.parent / "output"

# (stage_num, stage_name, hw_badge, [(notebook_name, [prereqs])])
STAGE_DEFS = [
    (1, "Data Preparation", "CPU", [
        ("exploration_eos",      [DATA_DIR / "EOS-04_datasheet.xlsx"]),
        ("exploration_sentinel", [DATA_DIR / "sentinel-1.xlsx"]),
    ]),
    (2, "Classification", "CPU + XGB", [
        ("classification_censored",   []),
        ("classification_uncensored", []),
    ]),
    (3, "Classical ML", "GPU", [
        ("classical_ml_censored",   []),
        ("classical_ml_uncensored", []),
    ]),
    (4, "ANN Regression", "GPU", [
        ("ann_censored",   []),
        ("ann_uncensored", []),
    ]),
    (5, "PI Estimation", "GPU", [
        ("pi_estimation_censored",   []),
        ("pi_estimation_uncensored", []),
    ]),
    (6, "Conformal Regression", "CPU", [
        ("conformal_regression_censored",   []),
        ("conformal_regression_uncensored", []),
    ]),
    (7, "Advanced PI Methods", "GPU", [
        ("conformalized_quantile_regression_uncensored", []),
        ("quantile_regression_tau_tuning_uncensored",    []),
    ]),
    (8, "Quantile SVR Tuning", "CPU", [
        ("quantile_svr_HP_tuning", []),
    ]),
]

# Flat ordered list — preserves execution order across stages
NOTEBOOKS = [(n, p) for _, _, _, nbs in STAGE_DEFS for n, p in nbs]

# Notebook name → (stage_num, stage_name, hw_badge)
NB_TO_STAGE: dict[str, tuple[int, str, str]] = {
    n: (snum, sname, hw)
    for snum, sname, hw, nbs in STAGE_DEFS
    for n, _ in nbs
}

PRIOR_ESTIMATES = {
    "exploration_eos":                                90,
    "exploration_sentinel":                           90,
    "classification_censored":                       120,
    "classification_uncensored":                     120,
    "classical_ml_censored":                         180,
    "classical_ml_uncensored":                       180,
    "ann_censored":                                  300,
    "ann_uncensored":                                300,
    "pi_estimation_censored":                        360,
    "pi_estimation_uncensored":                      360,
    "conformal_regression_censored":                 150,
    "conformal_regression_uncensored":               150,
    "conformalized_quantile_regression_uncensored":  240,
    "quantile_regression_tau_tuning_uncensored":     240,
    "quantile_svr_HP_tuning":                       3600,
}

STATUS_STYLE = {
    "pending": ("dim white",  "○"),
    "skipped": ("yellow",     "⏭"),
    "running": ("bold cyan",  "◉"),
    "passed":  ("bold green", "✓"),
    "failed":  ("bold red",   "✗"),
}

console = Console()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class NotebookState:
    def __init__(self, name: str, prereqs: list):
        self.name       = name
        self.prereqs    = prereqs
        self.status     = "pending"
        self.elapsed    = 0.0
        self.error_tail = ""

    @property
    def display_name(self) -> str:
        return self.name.replace("_", " ")


# ---------------------------------------------------------------------------
# ETA
# ---------------------------------------------------------------------------

def compute_eta(states: list[NotebookState]) -> float:
    completed = [(s.name, s.elapsed) for s in states if s.status == "passed"]
    scale = 1.0
    if completed:
        ratios = [e / PRIOR_ESTIMATES.get(n, 180) for n, e in completed if PRIOR_ESTIMATES.get(n, 180) > 0]
        if ratios:
            scale = sum(ratios) / len(ratios)
    return sum(
        PRIOR_ESTIMATES.get(s.name, 180) * scale
        for s in states if s.status in ("pending", "running")
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def build_display(states: list[NotebookState], overall: Progress,
                  current_start: Optional[float]) -> Panel:
    now = time.time()

    table = Table(box=box.ROUNDED, expand=True, show_header=True,
                  header_style="bold white on #1a1a2e", border_style="bright_black")
    table.add_column("#",        width=3,  justify="right", style="dim")
    table.add_column("Notebook", min_width=50)
    table.add_column("Status",   width=11, justify="center")
    table.add_column("Duration", width=10, justify="right")

    seen_stage: Optional[int] = None
    for i, s in enumerate(states, 1):
        snum, sname, hw = NB_TO_STAGE[s.name]

        # Stage header row on stage change
        if snum != seen_stage:
            seen_stage = snum
            gpu_tag = "[bold yellow]⚡[/]" if "GPU" in hw else "[dim]  [/]"
            table.add_row(
                "", f"  {gpu_tag} [bold bright_black]Stage {snum} — {sname}[/]  [dim]{hw}[/]",
                "", "", style="on #0d1117",
            )

        style_color, icon = STATUS_STYLE[s.status]
        if s.status == "running" and current_start:
            dur_str = _fmt_time(now - current_start) + "…"
        elif s.status in ("passed", "failed"):
            dur_str = _fmt_time(s.elapsed)
        else:
            dur_str = "—"

        table.add_row(str(i), s.display_name,
                      Text(f"{icon} {s.status.upper()}", style=style_color), dur_str)

    # Summary bar
    done    = sum(1 for s in states if s.status in ("passed", "skipped", "failed"))
    passed  = sum(1 for s in states if s.status == "passed")
    failed  = sum(1 for s in states if s.status == "failed")
    skipped = sum(1 for s in states if s.status == "skipped")
    eta_str = _fmt_time(compute_eta(states))

    summary = (
        f"  [green]✓ {passed} passed[/]  [red]✗ {failed} failed[/]  "
        f"[yellow]⏭ {skipped} skipped[/]  [dim]○ {len(states) - done} pending[/]   "
        f"[bold white]ETA {eta_str}[/]"
    )

    layout = Table.grid(expand=True)
    layout.add_column()
    layout.add_row(overall)
    layout.add_row(Text.from_markup(summary))
    layout.add_row(Text(""))
    layout.add_row(table)

    return Panel(layout, title="[bold white] SAR Soil Moisture — Experiment Pipeline [/]",
                 border_style="bright_blue", padding=(0, 1))


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


# ---------------------------------------------------------------------------
# GPU library path
# ---------------------------------------------------------------------------

def _build_ld_library_path() -> str:
    """Prepend venv NVIDIA wheel paths so TF 2.21 picks up cuDNN 9.3."""
    import site
    sp = Path(site.getsitepackages()[0])
    dirs = [
        str(sp / "nvidia" / "cudnn" / "lib"),
        str(sp / "nvidia" / "cublas" / "lib"),
        "/usr/lib/x86_64-linux-gnu",
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        dirs.append(existing)
    return ":".join(p for p in dict.fromkeys(dirs) if p)  # deduplicate, preserve order


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_notebook(state: NotebookState, env: dict) -> bool:
    nb_path = CODE_DIR / f"{state.name}.ipynb"
    proc = subprocess.run(
        [sys.executable, "-m", "nbconvert",
         "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=7200",
         "--ExecutePreprocessor.kernel_name=python3",
         str(nb_path)],
        capture_output=True, text=True,
        cwd=str(CODE_DIR), env=env,
    )
    if proc.returncode != 0:
        state.error_tail = (proc.stderr or proc.stdout or "")[-1000:].strip()
    return proc.returncode == 0


def _handle_failure(state: NotebookState, live: Live) -> bool:
    """Print failure info and ask whether to continue. Returns True = keep going."""
    live.stop()
    console.print(f"\n[bold red]✗ FAILED:[/] {state.name}")
    snum, sname, _ = NB_TO_STAGE[state.name]
    console.print(f"  [dim]Stage {snum} — {sname}[/]")
    if state.error_tail:
        console.print(f"[dim]{state.error_tail}[/]")
    if sys.stdin.isatty():
        try:
            console.print("\n[yellow]Continue with next notebook? [Y/n][/] ", end="")
            ans = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "y"
    else:
        ans = "y"
        console.print("[yellow]Non-interactive mode — continuing automatically.[/]")
    live.start()
    return ans != "n"


def run_pipeline(skip_names: list[str], start_from: Optional[str], export_after: bool):
    env = {**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path()}
    states = [NotebookState(name, prereqs) for name, prereqs in NOTEBOOKS]

    reached = (start_from is None)
    for s in states:
        if not reached:
            if s.name == start_from:
                reached = True
            else:
                s.status, s.elapsed = "skipped", 0.0
        if s.name in skip_names:
            s.status, s.elapsed = "skipped", 0.0

    overall = Progress(
        SpinnerColumn(),
        TextColumn("[bold white]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        expand=False,
    )
    task_id    = overall.add_task("Running notebooks", total=len(states))
    done_count = sum(1 for s in states if s.status == "skipped")
    overall.update(task_id, completed=done_count)

    current_start: Optional[float] = None
    pipeline_start = time.time()
    prev_stage: Optional[int] = None

    with Live(build_display(states, overall, None), refresh_per_second=4,
              console=console) as live:
        for s in states:
            snum, sname, _ = NB_TO_STAGE[s.name]

            # Stage boundary announcement
            if snum != prev_stage:
                prev_stage = snum
                if s.status != "skipped":
                    overall.update(task_id, description=f"Stage {snum}: {sname}")

            if s.status == "skipped":
                overall.update(task_id, advance=1)
                live.update(build_display(states, overall, None))
                continue

            # Check prereqs
            missing = [str(p) for p in s.prereqs if not Path(p).exists()]
            if missing:
                s.status, s.elapsed = "skipped", 0.0
                overall.update(task_id, advance=1,
                               description=f"Skipped {s.display_name} (missing prereqs)")
                live.update(build_display(states, overall, None))
                continue

            # Run
            s.status      = "running"
            current_start = time.time()
            overall.update(task_id, description=f"Stage {snum} › {s.display_name}")
            live.update(build_display(states, overall, current_start))

            ok = run_notebook(s, env)

            s.elapsed     = time.time() - current_start
            s.status      = "passed" if ok else "failed"
            current_start = None
            overall.update(task_id, advance=1,
                           description=f"{'✓' if ok else '✗'} {s.display_name}")
            live.update(build_display(states, overall, None))

            if not ok and not _handle_failure(s, live):
                break

    # ── Final summary ────────────────────────────────────────────────────────
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
            snum, sname, _ = NB_TO_STAGE[s.name]
            console.print(f"  • {s.name}  [dim](Stage {snum}: {sname})[/]")
        console.print(
            f"\n[dim]Re-run failed stage(s) with:[/] "
            f"[cyan]--from-stage {NB_TO_STAGE[failed[0].name][0]}[/]"
        )

    # Per-stage timing breakdown
    console.print()
    t = Table("Stage", "Notebook", "Status", "Duration", box=box.SIMPLE_HEAVY)
    prev = None
    for s in states:
        snum, sname, _ = NB_TO_STAGE[s.name]
        stage_label = f"Stage {snum}: {sname}" if snum != prev else ""
        prev = snum
        color = {"passed": "green", "failed": "red", "skipped": "yellow"}.get(s.status, "white")
        t.add_row(stage_label, s.name, f"[{color}]{s.status.upper()}[/]",
                  _fmt_time(s.elapsed) if s.elapsed else "—")
    console.print(t)

    if export_after or (not failed and _ask_export()):
        _run_export(env)


def _ask_export() -> bool:
    console.print("\n[bold white]Export results to Excel? [Y/n][/] ", end="")
    try:
        return input().strip().lower() != "n"
    except (EOFError, KeyboardInterrupt):
        return False


def _run_export(env: dict):
    console.print("\n[bold cyan]Exporting metrics + plots to Excel…[/]")
    start = time.time()
    proc = subprocess.run(
        [sys.executable, str(CODE_DIR / "export_to_excel.py")],
        cwd=str(CODE_DIR), env=env,
    )
    elapsed = time.time() - start
    if proc.returncode == 0:
        console.print(f"[green]✓ Export complete[/] ({_fmt_time(elapsed)})")
        console.print(f"  Results → [bold]{OUTPUT_DIR}/SAR_Moisture_Final_Report/[/]")
    else:
        console.print("[red]✗ Export failed[/]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    all_names   = [n for n, _ in NOTEBOOKS]
    stage_nums  = [s[0] for s in STAGE_DEFS]

    parser = argparse.ArgumentParser(
        description="Stage-wise SAR soil moisture experiment pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--from",       dest="start_from", metavar="NOTEBOOK",
                        help="Resume from a specific notebook (skip all before it)")
    parser.add_argument("--from-stage", dest="from_stage",  type=int, metavar="N",
                        help="Resume from stage N (skip all notebooks before that stage)")
    parser.add_argument("--stage",      dest="only_stage",  type=int, metavar="N",
                        help="Run only stage N (skip all other stages)")
    parser.add_argument("--skip",       dest="skip", nargs="+", metavar="NOTEBOOK", default=[],
                        help="Skip specific notebooks by name")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip all notebooks and only run Excel export")
    args = parser.parse_args()

    if args.export_only:
        env = {**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path()}
        _run_export(env)
        return

    # Validate args
    if args.start_from and args.start_from not in all_names:
        console.print(f"[red]Unknown notebook:[/] {args.start_from!r}")
        console.print(f"Valid names: {', '.join(all_names)}")
        sys.exit(1)

    if args.from_stage and args.from_stage not in stage_nums:
        console.print(f"[red]Invalid stage:[/] {args.from_stage}. Valid: {stage_nums}")
        sys.exit(1)

    if args.only_stage and args.only_stage not in stage_nums:
        console.print(f"[red]Invalid stage:[/] {args.only_stage}. Valid: {stage_nums}")
        sys.exit(1)

    for name in args.skip:
        if name not in all_names:
            console.print(f"[red]Unknown notebook to skip:[/] {name!r}")
            sys.exit(1)

    # Resolve start_from
    start_from = args.start_from

    if args.from_stage:
        # Find first notebook in that stage
        for _, _, _, nbs in STAGE_DEFS:
            if NB_TO_STAGE[nbs[0][0]][0] == args.from_stage:
                start_from = nbs[0][0]
                break

    # Resolve skip list for --stage
    skip_names = list(args.skip)
    if args.only_stage:
        skip_names += [
            n for n, _ in NOTEBOOKS
            if NB_TO_STAGE[n][0] != args.only_stage and n not in skip_names
        ]

    try:
        run_pipeline(
            skip_names=skip_names,
            start_from=start_from,
            export_after=False,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/]")


if __name__ == "__main__":
    main()
