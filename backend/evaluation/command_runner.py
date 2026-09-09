"""Neutral black-box command execution with redacted logs and resource sampling."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO, TypeVar

import psutil

from backend.evaluation.core_schemas import (
    CommandIsolationEvidence,
    CommandObservation,
    CommandResourceUsage,
)

NumericT = TypeVar("NumericT", int, float)


@dataclass(frozen=True)
class _ProcessSample:
    peak_rss_bytes: int | None
    user_cpu_seconds: float | None
    system_cpu_seconds: float | None
    read_bytes: int | None
    write_bytes: int | None


def run_observed_command(
    command: list[str],
    *,
    cwd: Path,
    log_dir: Path,
    label: str,
    repository_root: Path,
    redact_values: list[str] | None = None,
    sample_interval_seconds: float = 0.05,
    timeout_seconds: float | None = None,
    environment: dict[str, str] | None = None,
    execution_command: list[str] | None = None,
    isolation: CommandIsolationEvidence | None = None,
) -> CommandObservation:
    """Execute one argument-vector command without a shell.

    A configured timeout terminates the complete process group so child
    processes cannot outlive the evaluator.
    """

    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    resolved_cwd = cwd.resolve()
    if not resolved_cwd.is_dir():
        raise FileNotFoundError(
            f"Command working directory does not exist: {resolved_cwd}"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    started_wall = datetime.now(UTC)
    started = time.perf_counter()
    child_environment = os.environ.copy() if environment is None else environment.copy()
    child_environment.setdefault("PYTHONFAULTHANDLER", "1")
    actual_command = execution_command or command
    process = subprocess.Popen(
        actual_command,
        cwd=resolved_cwd,
        env=child_environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    assert process.stdout is not None
    assert process.stderr is not None
    readers = [
        threading.Thread(
            target=_read_stream,
            args=(process.stdout, stdout_lines),
            daemon=True,
        ),
        threading.Thread(
            target=_read_stream,
            args=(process.stderr, stderr_lines),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    tracked = psutil.Process(process.pid)
    peak_rss: int | None = None
    max_user_cpu: float | None = None
    max_system_cpu: float | None = None
    max_read_bytes: int | None = None
    max_write_bytes: int | None = None
    timed_out = False
    while process.poll() is None:
        sample = _sample_process_tree(tracked)
        peak_rss = _max_optional(peak_rss, sample.peak_rss_bytes)
        max_user_cpu = _max_optional(max_user_cpu, sample.user_cpu_seconds)
        max_system_cpu = _max_optional(max_system_cpu, sample.system_cpu_seconds)
        max_read_bytes = _max_optional(max_read_bytes, sample.read_bytes)
        max_write_bytes = _max_optional(max_write_bytes, sample.write_bytes)
        if (
            timeout_seconds is not None
            and time.perf_counter() - started >= timeout_seconds
        ):
            timed_out = True
            _terminate_process_group(process)
            break
        time.sleep(sample_interval_seconds)

    final_sample = _sample_process_tree(tracked)
    peak_rss = _max_optional(peak_rss, final_sample.peak_rss_bytes)
    max_user_cpu = _max_optional(max_user_cpu, final_sample.user_cpu_seconds)
    max_system_cpu = _max_optional(max_system_cpu, final_sample.system_cpu_seconds)
    max_read_bytes = _max_optional(max_read_bytes, final_sample.read_bytes)
    max_write_bytes = _max_optional(max_write_bytes, final_sample.write_bytes)
    exit_code = process.wait()
    for reader in readers:
        reader.join()

    completed_wall = datetime.now(UTC)
    duration = time.perf_counter() - started
    values = [value for value in (redact_values or []) if len(value) >= 4]
    raw_stdout = "".join(stdout_lines)
    raw_stderr = "".join(stderr_lines)
    redactions_applied = sum(
        raw_stdout.count(value) + raw_stderr.count(value) for value in set(values)
    )
    stdout = _redact(raw_stdout, values)
    stderr = _redact(raw_stderr, values)
    stdout_path = log_dir / f"{label}.stdout.log"
    stderr_path = log_dir / f"{label}.stderr.log"
    _write_atomic(stdout_path, stdout)
    _write_atomic(stderr_path, stderr)

    return CommandObservation(
        command=command,
        executed_command=actual_command,
        cwd=_display_path(resolved_cwd, repository_root),
        started_at=started_wall.isoformat(),
        completed_at=completed_wall.isoformat(),
        duration_seconds=duration,
        exit_code=exit_code,
        timed_out=timed_out,
        stdout_log=_display_path(stdout_path, repository_root),
        stderr_log=_display_path(stderr_path, repository_root),
        stdout_tail=stdout[-8_000:],
        stderr_tail=stderr[-8_000:],
        redactions_applied=redactions_applied,
        environment_variable_names=sorted(child_environment),
        isolation=isolation,
        resources=CommandResourceUsage(
            peak_rss_bytes=peak_rss,
            user_cpu_seconds=max_user_cpu,
            system_cpu_seconds=max_system_cpu,
            read_bytes=max_read_bytes,
            write_bytes=max_write_bytes,
            sample_interval_seconds=sample_interval_seconds,
        ),
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate a timed-out command and every descendant, then force-kill."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _read_stream(stream: TextIO, destination: list[str]) -> None:
    try:
        for line in iter(stream.readline, ""):
            destination.append(line)
    finally:
        stream.close()


def _sample_process_tree(process: psutil.Process) -> _ProcessSample:
    try:
        processes = [process, *process.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        processes = []

    rss: int | None = None
    user_cpu: float | None = None
    system_cpu: float | None = None
    read_bytes: int | None = None
    write_bytes: int | None = None
    for item in processes:
        try:
            memory = item.memory_info()
            cpu = item.cpu_times()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        rss = (rss or 0) + int(memory.rss)
        user_cpu = (user_cpu or 0.0) + float(cpu.user)
        system_cpu = (system_cpu or 0.0) + float(cpu.system)
        try:
            io = item.io_counters()
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue
        read_bytes = (read_bytes or 0) + int(io.read_bytes)
        write_bytes = (write_bytes or 0) + int(io.write_bytes)
    return _ProcessSample(rss, user_cpu, system_cpu, read_bytes, write_bytes)


def _max_optional(
    current: NumericT | None, candidate: NumericT | None
) -> NumericT | None:
    if candidate is None:
        return current
    return candidate if current is None else max(current, candidate)


def _redact(text: str, values: list[str]) -> str:
    for value in sorted(set(values), key=len, reverse=True):
        text = text.replace(value, "<redacted>")
    return text


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
