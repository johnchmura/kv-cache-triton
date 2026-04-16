"""Per-session artifacts under tests/logs/<timestamp>/ (CSV, plots, summary)."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

_SESSION_CONFIG: pytest.Config | None = None


def _get_config() -> pytest.Config:
    assert _SESSION_CONFIG is not None
    return _SESSION_CONFIG


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_rss_bytes_linux() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def pytest_sessionstart(session: pytest.Session) -> None:
    global _SESSION_CONFIG
    _SESSION_CONFIG = session.config
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_root = _project_root() / "tests" / "logs" / ts
    log_root.mkdir(parents=True, exist_ok=True)
    session.config._kv_log_dir = log_root
    session.config._kv_session_t0 = time.perf_counter()
    session.config._kv_metrics_by_nodeid = {}
    session.config._kv_call_info = {}
    session.config._kv_result_rows: list[dict] = []


@pytest.fixture(autouse=True)
def _kv_resource_metrics(request: pytest.FixtureRequest) -> None:
    rss_before = _read_rss_bytes_linux()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    yield

    nodeid = request.node.nodeid
    cfg = request.config
    if not hasattr(cfg, "_kv_metrics_by_nodeid"):
        return

    cuda_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    rss_after = _read_rss_bytes_linux()
    rss_delta = None
    if rss_before is not None and rss_after is not None:
        rss_delta = rss_after - rss_before

    cfg._kv_metrics_by_nodeid[nodeid] = {
        "cuda_peak_alloc_bytes": cuda_peak,
        "process_rss_delta_bytes": rss_delta,
    }


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _SESSION_CONFIG is None:
        return
    cfg = _SESSION_CONFIG
    nodeid = report.nodeid

    if report.when == "call":
        err = ""
        if report.failed and report.longreprtext:
            err = report.longreprtext.replace("\n", " ")[:500]
        duration = getattr(report, "duration", 0.0) or 0.0
        cfg._kv_call_info[nodeid] = {
            "outcome": report.outcome,
            "duration_s": duration,
            "error_summary": err,
        }
        return

    if report.when != "teardown":
        return

    if not hasattr(cfg, "_kv_result_rows"):
        return

    call_info = cfg._kv_call_info.pop(nodeid, None)
    if call_info is None:
        return

    metrics = cfg._kv_metrics_by_nodeid.pop(nodeid, {})
    cuda_peak = metrics.get("cuda_peak_alloc_bytes")
    rss_delta = metrics.get("process_rss_delta_bytes")

    cfg._kv_result_rows.append(
        {
            "nodeid": nodeid,
            "outcome": call_info["outcome"],
            "duration_s": call_info["duration_s"],
            "cuda_peak_alloc_bytes": cuda_peak if cuda_peak is not None else "",
            "process_rss_delta_bytes": rss_delta if rss_delta is not None else "",
            "error_summary": call_info["error_summary"],
        }
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    cfg = session.config
    log_dir = getattr(cfg, "_kv_log_dir", None)
    if log_dir is None:
        return

    leftover = getattr(cfg, "_kv_call_info", None) or {}
    for nodeid, info in list(leftover.items()):
        metrics = cfg._kv_metrics_by_nodeid.pop(nodeid, {})
        cuda_peak = metrics.get("cuda_peak_alloc_bytes")
        rss_delta = metrics.get("process_rss_delta_bytes")
        cfg._kv_result_rows.append(
            {
                "nodeid": nodeid,
                "outcome": info["outcome"],
                "duration_s": info["duration_s"],
                "cuda_peak_alloc_bytes": cuda_peak if cuda_peak is not None else "",
                "process_rss_delta_bytes": rss_delta if rss_delta is not None else "",
                "error_summary": info["error_summary"],
            }
        )
    cfg._kv_call_info = {}

    rows = getattr(cfg, "_kv_result_rows", [])
    t0 = getattr(cfg, "_kv_session_t0", None)
    elapsed = (time.perf_counter() - t0) if t0 is not None else None

    csv_path = log_dir / "results.csv"
    fieldnames = [
        "nodeid",
        "outcome",
        "duration_s",
        "cuda_peak_alloc_bytes",
        "process_rss_delta_bytes",
        "error_summary",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    cuda_name = None
    if torch.cuda.is_available():
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_name = "unknown"

    summary = {
        "exitstatus": exitstatus,
        "elapsed_s": elapsed,
        "pytest_version": pytest.__version__,
        "cuda_device_name": cuda_name,
    }
    with open(log_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not rows:
        return

    labels = [r["nodeid"].split("::")[-1][:40] for r in rows]
    durations = [float(r["duration_s"]) for r in rows]
    outcomes = [r["outcome"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.25 * len(rows))))
    ax.barh(range(len(rows)), durations)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("duration_s")
    ax.set_title("Test duration")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(log_dir / "duration_by_test.png", dpi=120)
    plt.close(fig)

    cuda_vals = []
    for r in rows:
        v = r["cuda_peak_alloc_bytes"]
        if v == "" or v is None:
            cuda_vals.append(0.0)
        else:
            cuda_vals.append(float(v))

    if any(cuda_vals):
        fig2, ax2 = plt.subplots(figsize=(10, max(3, 0.25 * len(rows))))
        ax2.barh(range(len(rows)), [x / (1024**2) for x in cuda_vals])
        ax2.set_yticks(range(len(rows)))
        ax2.set_yticklabels(labels, fontsize=7)
        ax2.set_xlabel("CUDA peak alloc (MiB)")
        ax2.set_title("CUDA peak memory allocated during test")
        ax2.invert_yaxis()
        fig2.tight_layout()
        fig2.savefig(log_dir / "cuda_peak_by_test.png", dpi=120)
        plt.close(fig2)

    has_pair = [
        (float(r["duration_s"]), float(r["cuda_peak_alloc_bytes"]) / (1024**2))
        for r in rows
        if r["cuda_peak_alloc_bytes"] not in ("", None)
    ]
    if len(has_pair) >= 2:
        ds, ms = zip(*has_pair)
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.scatter(ds, ms, alpha=0.7)
        ax3.set_xlabel("duration_s")
        ax3.set_ylabel("CUDA peak (MiB)")
        ax3.set_title("Duration vs CUDA peak memory")
        fig3.tight_layout()
        fig3.savefig(log_dir / "duration_vs_cuda_peak.png", dpi=120)
        plt.close(fig3)

    c = Counter(outcomes)
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    ax4.bar(list(c.keys()), list(c.values()))
    ax4.set_ylabel("count")
    ax4.set_title("Test outcomes")
    fig4.tight_layout()
    fig4.savefig(log_dir / "outcome_counts.png", dpi=120)
    plt.close(fig4)
