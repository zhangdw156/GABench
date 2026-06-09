#!/usr/bin/env python3
"""Estimate GABench evaluation progress with the runner's resume rule.

Progress is measured the same way as ``runners/run_benchmark.py`` resumes work:
a task is counted as completed only when at least one record for that task has
``status == "success"``. Error records stay pending/retryable.

Examples:
    uv run python scripts/estimate_eval_progress.py
    uv run python scripts/estimate_eval_progress.py --model qwen3-4b-instruct-2507 --agent react
    uv run python scripts/estimate_eval_progress.py --run results/qwen3-4b-instruct-2507/react/all_tools_20260609_120000_parallel.jsonl
    uv run python scripts/estimate_eval_progress.py --json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = REPO_ROOT / "benchmark" / "benchmark.csv"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


@dataclass(frozen=True)
class RunProgress:
    result_path: str
    run_id: str
    model: str | None
    agent: str | None
    completed: int
    retryable: int
    pending: int
    total: int
    progress: float
    records: int
    success_records: int
    error_records: int
    duplicate_task_records: int
    malformed_lines: int


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_task_ids(benchmark_path: Path) -> list[str]:
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"benchmark file not found: {benchmark_path}; run `uv run scripts/download_data.py` first"
        )
    ids: list[str] = []
    with benchmark_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            task_id = row.get("ID") or row.get("id") or row.get("任务ID")
            if task_id is None:
                continue
            task_id = str(task_id).strip()
            if task_id:
                ids.append(task_id)
    if not ids:
        raise ValueError(f"no task IDs found in benchmark file: {benchmark_path}")
    return list(dict.fromkeys(ids))


def discover_result_files(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    return sorted(path for path in results_dir.rglob("*.jsonl") if path.is_file())


def find_run_path(run: str, results_dir: Path) -> Path:
    run_path = Path(run)
    if run_path.exists():
        return run_path
    if not run_path.is_absolute():
        candidate = REPO_ROOT / run_path
        if candidate.exists():
            return candidate
    matches = sorted(results_dir.rglob(f"{run}.jsonl"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"run not found as a file or run id under {results_dir}: {run}")
    raise ValueError(f"run id is ambiguous under {results_dir}: {run}\n" + "\n".join(map(str, matches)))


def infer_model_agent(path: Path, records: Iterable[dict[str, Any]]) -> tuple[str | None, str | None]:
    parts = path.resolve().parts
    model = None
    agent = None
    if "results" in parts:
        idx = len(parts) - 1 - parts[::-1].index("results")
        if len(parts) > idx + 3 and parts[idx + 1] != "debug":
            model = parts[idx + 1]
            agent = parts[idx + 2]

    if model and agent:
        return model, agent

    for record in records:
        if not isinstance(record, dict):
            continue
        model = model or (str(record.get("model")) if record.get("model") is not None else None)
        agent = agent or (str(record.get("agent")) if record.get("agent") is not None else None)
        if model and agent:
            break
    return model, agent


def progress_for_file(path: Path, all_task_ids: set[str]) -> RunProgress:
    records: list[dict[str, Any]] = []
    task_record_counts: dict[str, int] = {}
    successful_tasks: set[str] = set()
    latest_status_by_task: dict[str, str | None] = {}
    malformed_lines = 0
    success_records = 0
    error_records = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            if not isinstance(record, dict):
                malformed_lines += 1
                continue
            records.append(record)

            task_id_raw = record.get("task_id")
            if task_id_raw is None:
                continue
            task_id = str(task_id_raw)
            status = record.get("status")
            latest_status_by_task[task_id] = str(status) if status is not None else None
            task_record_counts[task_id] = task_record_counts.get(task_id, 0) + 1
            if status == "success":
                successful_tasks.add(task_id)
                success_records += 1
            elif status == "error":
                error_records += 1

    model, agent = infer_model_agent(path, records)
    completed = len(successful_tasks & all_task_ids)
    retryable = sum(
        1
        for task_id, status in latest_status_by_task.items()
        if task_id in all_task_ids and task_id not in successful_tasks and status == "error"
    )
    total = len(all_task_ids)
    pending = max(total - completed, 0)
    duplicate_task_records = sum(max(count - 1, 0) for task_id, count in task_record_counts.items() if task_id in all_task_ids)

    return RunProgress(
        result_path=relpath(path),
        run_id=path.stem,
        model=model,
        agent=agent,
        completed=completed,
        retryable=retryable,
        pending=pending,
        total=total,
        progress=completed / total if total else 0.0,
        records=len(records),
        success_records=success_records,
        error_records=error_records,
        duplicate_task_records=duplicate_task_records,
        malformed_lines=malformed_lines,
    )


def select_runs(
    paths: list[Path],
    *,
    model: str | None,
    agent: str | None,
    all_task_ids: set[str],
) -> list[RunProgress]:
    runs = [progress_for_file(path, all_task_ids) for path in paths]
    if model:
        runs = [run for run in runs if run.model == model or f"/results/{model}/" in f"/{run.result_path}"]
    if agent:
        runs = [run for run in runs if run.agent == agent or f"/{agent}/" in f"/{run.result_path}"]
    return runs


def pct(value: float) -> str:
    return f"{value * 100:6.2f}%"


def print_table(runs: list[RunProgress]) -> None:
    if not runs:
        print("No result JSONL files found for the selected filters.")
        return

    headers = [
        "run",
        "model",
        "agent",
        "done/total",
        "progress",
        "retryable",
        "records",
        "dup",
        "bad",
        "path",
    ]
    rows = []
    for run in sorted(runs, key=lambda item: (item.model or "", item.agent or "", item.result_path)):
        rows.append(
            [
                run.run_id,
                run.model or "-",
                run.agent or "-",
                f"{run.completed}/{run.total}",
                pct(run.progress),
                str(run.retryable),
                str(run.records),
                str(run.duplicate_task_records),
                str(run.malformed_lines),
                run.result_path,
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt.format(*row))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK, help="Path to benchmark.csv")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory containing result JSONL files")
    parser.add_argument("--run", help="Specific result JSONL path or run id")
    parser.add_argument("--model", help="Filter by model")
    parser.add_argument("--agent", help="Filter by agent")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    benchmark_path = args.benchmark if args.benchmark.is_absolute() else (REPO_ROOT / args.benchmark)
    results_dir = args.results_dir if args.results_dir.is_absolute() else (REPO_ROOT / args.results_dir)

    task_ids = load_task_ids(benchmark_path)
    task_id_set = set(task_ids)

    if args.run:
        paths = [find_run_path(args.run, results_dir)]
    else:
        paths = discover_result_files(results_dir)

    runs = select_runs(paths, model=args.model, agent=args.agent, all_task_ids=task_id_set)

    if args.json:
        print(json.dumps([asdict(run) for run in runs], ensure_ascii=False, indent=2))
    else:
        print(f"Benchmark tasks: {len(task_ids)} ({relpath(benchmark_path)})")
        print_table(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
