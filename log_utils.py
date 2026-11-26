"""
Shared console-to-file logging utility for the XAI-IDS pipeline.

Mirrors stdout and stderr to:
  log_output/<log_name>/<log_name>.txt                (latest run)
  log_output/<log_name>/history/<log_name>_<ts>.txt   (archived run for comparison)
"""

import atexit
import datetime
import os
import sys
import time


class _TeeStream:
    """Write stream output to both the original terminal stream and log files."""

    def __init__(self, orig_stream, log_files):
        self._orig = orig_stream
        self._files = log_files

    def write(self, data):
        if not data:
            return 0
        try:
            self._orig.write(data)
        except UnicodeEncodeError:
            enc = getattr(self._orig, "encoding", None) or "ascii"
            safe = data.encode(enc, errors="replace").decode(enc, errors="replace")
            self._orig.write(safe)
        for fh in self._files:
            if not fh.closed:
                fh.write(data)
                fh.flush()
        return len(data)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        for fh in self._files:
            if not fh.closed:
                fh.flush()

    def isatty(self):
        return hasattr(self._orig, "isatty") and self._orig.isatty()

    @property
    def encoding(self):
        return getattr(self._orig, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self._orig, name)


_LOG_ACTIVE = False


def setup_run_logger(log_name, script_path=None, log_root="log_output"):
    """
    Redirect stdout & stderr so every run is logged to:
      <log_root>/<log_name>/<log_name>.txt
    and archived to:
      <log_root>/<log_name>/history/<log_name>_<YYYYMMDD_HHMMSS>.txt
    """
    global _LOG_ACTIVE
    if _LOG_ACTIVE:
        return None

    log_dir = os.path.join(log_root, log_name)
    history_dir = os.path.join(log_dir, "history")
    os.makedirs(history_dir, exist_ok=True)

    now = datetime.datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    latest_path = os.path.join(log_dir, f"{log_name}.txt")
    archive_path = os.path.join(history_dir, f"{log_name}_{ts}.txt")

    latest_fh = open(latest_path, "w", encoding="utf-8", errors="replace")
    archive_fh = open(archive_path, "w", encoding="utf-8", errors="replace")
    log_files = [latest_fh, archive_fh]

    script_label = os.path.basename(script_path) if script_path else log_name
    header = (
        f"{'=' * 70}\n"
        f"  RUN LOG       : {log_name}\n"
        f"  Script        : {script_label}\n"
        f"  Start Time    : {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Working Dir   : {os.getcwd()}\n"
        f"  Latest Log    : {latest_path}\n"
        f"  Archived Log  : {archive_path}\n"
        f"{'=' * 70}\n\n"
    )
    for fh in log_files:
        fh.write(header)
        fh.flush()

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _TeeStream(orig_stdout, log_files)
    sys.stderr = _TeeStream(orig_stderr, log_files)
    _LOG_ACTIVE = True
    t0 = time.time()

    def _close_logs():
        global _LOG_ACTIVE
        end_now = datetime.datetime.now()
        elapsed = time.time() - t0
        footer = (
            f"\n{'=' * 70}\n"
            f"  End Time      : {end_now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Elapsed       : {elapsed:.2f}s\n"
            f"{'=' * 70}\n"
        )
        for fh in log_files:
            if not fh.closed:
                fh.write(footer)
                fh.flush()
                fh.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        _LOG_ACTIVE = False

    atexit.register(_close_logs)
    return latest_path
