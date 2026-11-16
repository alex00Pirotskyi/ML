"""Shared CPU budget for V5 labeling and training (no CLI settings).

Aim to use ~80% of *available* CPUs, respecting Linux affinity and cgroup
CPU quotas. This is a worker/thread budget, not an exact process % throttle.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

CPU_FRACTION = 0.80


def available_cpus() -> int:
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    # Cgroup v2 quota; important for Docker/server jobs that expose more
    # logical CPUs through affinity than their actual CPU entitlement.
    path = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota, period = path.read_text().strip().split()[:2]
        if quota != "max" and int(period) > 0 and int(quota) > 0:
            affinity = min(affinity, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError, IndexError):
        pass
    # Cgroup v1, if applicable.
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0 and p > 0:
            affinity = min(affinity, max(1, math.ceil(q / p)))
    except (OSError, ValueError):
        pass
    return max(1, int(affinity))


def cpu_workers(fraction: float = CPU_FRACTION) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("CPU_FRACTION must be in (0, 1]")
    return max(1, math.floor(available_cpus() * fraction))


def prepare_process_worker() -> None:
    """Prevent process-pool oversubscription by OpenMP / BLAS backends."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass
