from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import threading
from typing import Callable, Iterable


_EXECUTOR_LOCK = threading.Lock()

# A shared executor is reused across frames to avoid thread churn.
_CURRENT_EXECUTOR: ThreadPoolExecutor | None = None
_CURRENT_MAX_WORKERS: int = 0


def _get_max_workers_cap() -> int:
    """Upper bound on updater worker threads.

    Why: Having `max_workers == number_of_updaters` can easily create hundreds
    of OS threads and causes severe stutter (oversubscription), especially on
    Windows. Default leaves one core for the main OpenGL/event thread.

    Override via env var: `MANIM_UPDATER_MAX_WORKERS`.
    """
    env = os.environ.get("MANIM_UPDATER_MAX_WORKERS")
    if env:
        try:
            value = int(env)
        except ValueError:
            value = 0
        if value > 0:
            return value

    cpu = os.cpu_count() or 4
    if cpu <= 1:
        return 1
    return max(1, min(32, cpu - 1))

# Thread-local marker to avoid recursive parallel submission, which can
# deadlock when a worker thread tries to synchronously wait on the same pool.
_TLS = threading.local()


def _is_inside_parallel_updater() -> bool:
    return bool(getattr(_TLS, "in_parallel_updater", False))


def _mark_parallel_updater(active: bool) -> None:
    _TLS.in_parallel_updater = active


def _acquire_executor(_: int) -> ThreadPoolExecutor:
    """Return the shared executor (created lazily).

    Note: we intentionally do NOT size the pool to the number of updaters.
    Oversubscription causes frame pacing jitter and UI stutter.
    """
    global _CURRENT_EXECUTOR, _CURRENT_MAX_WORKERS

    target = _get_max_workers_cap()
    with _EXECUTOR_LOCK:
        if _CURRENT_EXECUTOR is None or _CURRENT_MAX_WORKERS != target:
            _CURRENT_MAX_WORKERS = target
            _CURRENT_EXECUTOR = ThreadPoolExecutor(
                max_workers=_CURRENT_MAX_WORKERS,
                thread_name_prefix="ManimUpdater",
            )
        return _CURRENT_EXECUTOR


def run_updaters_in_parallel(
    callables: Iterable[Callable[[], object]],
    *,
    min_workers: int,
) -> None:
    """Run a batch of callables concurrently and wait for all to finish."""
    if _is_inside_parallel_updater():
        for func in callables:
            func()
        return

    executor = _acquire_executor(min_workers)
    futures = [executor.submit(_run_marked, func) for func in callables]
    first_exc: BaseException | None = None
    for fut in futures:
        try:
            fut.result()
        except BaseException as exc:
            if first_exc is None:
                first_exc = exc
    if first_exc is not None:
        raise first_exc


def _run_marked(func: Callable[[], object]) -> object:
    _mark_parallel_updater(True)
    try:
        return func()
    finally:
        _mark_parallel_updater(False)
