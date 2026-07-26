"""
Gymnasium Environment built around the PCSE library for crop simulation
Gym:  https://github.com/Farama-Foundation/Gymnasium
PCSE: https://github.com/ajwdewit/pcse

Based on the PCSE-Gym environment built by Hiske Overweg (https://github.com/WUR-AI/crop-gym)
Author: Collins Patrick Ohagwu

Optional, explicitly-called workaround for a Windows-only PCSE logging quirk.
"""

from __future__ import annotations

import logging
import logging.handlers


def silence_pcse_log_rotation() -> None:
    """Replace PCSE's root ``RotatingFileHandler`` with a plain, non-rotating one.

    Importing ``pcse`` (a side effect of building any ``Lintul3Env``) configures a
    shared root-logger ``RotatingFileHandler`` writing to ``~/.pcse/logs/pcse.log``.
    On Windows, if more than one process has ever imported ``pcse`` (e.g. two Jupyter
    kernels), that handler's log-rotation rename can fail with a ``PermissionError`` --
    caught internally by Python's own ``logging`` module, which prints a harmless but
    noisy "--- Logging error ---" banner instead of raising (the simulation itself is
    unaffected either way; only the one log line that triggered the failed rotation is
    lost).

    Call this once, after ``pcse`` would otherwise be imported (this function imports
    it itself if needed, to guarantee its logging config has already run), to replace
    the rotating handler with one that never rotates and so never attempts the failing
    rename. This is never called automatically by ``lintul3_gym`` -- only call it if
    the banner bothers you; PCSE's own file logging otherwise behaves exactly as it
    documents. Safe to call more than once: a no-op after the first call, since no
    ``RotatingFileHandler`` remains on the root logger to replace.
    """
    import pcse  # noqa: F401  # ensure pcse's dictConfig has installed its handlers

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            root_logger.removeHandler(handler)
            handler.close()
            replacement = logging.FileHandler(handler.baseFilename, mode="a", encoding="utf8", delay=True)
            replacement.setLevel(handler.level)
            if handler.formatter is not None:
                replacement.setFormatter(handler.formatter)
            root_logger.addHandler(replacement)
