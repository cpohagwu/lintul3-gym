"""Tests for the opt-in PCSE log-rotation workaround."""

import logging
import logging.handlers

import pytest

pytest.importorskip("pcse")

from lintul3_gym.logging_utils import silence_pcse_log_rotation


@pytest.fixture
def rotating_root_handler(tmp_path):
    """Attach a throwaway RotatingFileHandler to the root logger, then clean it up."""
    handler = logging.handlers.RotatingFileHandler(
        tmp_path / "test.log", maxBytes=1024, backupCount=1, delay=True
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield handler
    finally:
        root_logger.removeHandler(handler)
        handler.close()


def test_replaces_rotating_handler_with_plain_file_handler(rotating_root_handler) -> None:
    """The RotatingFileHandler is swapped for a non-rotating FileHandler on the same file."""
    silence_pcse_log_rotation()
    root_logger = logging.getLogger()

    assert not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root_logger.handlers)
    replacements = [
        h for h in root_logger.handlers
        if isinstance(h, logging.FileHandler) and h.baseFilename == rotating_root_handler.baseFilename
    ]
    assert len(replacements) == 1
    root_logger.removeHandler(replacements[0])
    replacements[0].close()


def test_is_idempotent_when_no_rotating_handler_remains() -> None:
    """Calling it again with nothing left to replace does not raise or duplicate handlers."""
    before = list(logging.getLogger().handlers)
    silence_pcse_log_rotation()
    silence_pcse_log_rotation()
    after = list(logging.getLogger().handlers)
    assert before == after
