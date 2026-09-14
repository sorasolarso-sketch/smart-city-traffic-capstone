"""Shared logging configuration for the capstone.

Importing this module has NO side effects. Nothing is configured until an
entry-point script calls :func:`configure_logging` from inside its own
``main()``. That is deliberate: library modules in this project obtain a
logger with ``logging.getLogger(__name__)`` and attach no handlers, so
importing them never hijacks another application's logging setup.

The formatter records, in order: timestamp, log level, module name and
message, which is the minimum the capstone brief requires.

Log levels as used across this project
--------------------------------------
DEBUG
    Fine-grained internal values that exist only to explain how a result
    was reached: quartile cut-points, per-month medians, scaler
    parameters. Suppressed on a normal run.
INFO
    Expected milestones: a file loaded, a stage completed, a figure or
    model written to disk, a CLI command invoked.
WARNING
    Recoverable but material events that change the data: rows dropped,
    values imputed, outliers capped, a monitoring threshold breached.
    Every warning states how many rows were affected and why.
ERROR
    A failure that stops the stage from completing. Always logged with
    exception detail attached, immediately before a graceful exit.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "pipeline.log"


def configure_logging(
    log_file: Path | str | None = DEFAULT_LOG_FILE,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    debug: bool = False,
    append: bool = True,
) -> logging.Logger:
    """Configure root logging for an entry-point script.

    Two handlers are attached to the root logger:

    * a ``StreamHandler`` writing to stdout at ``console_level``
    * a ``FileHandler`` writing to ``log_file`` at ``file_level``

    The file handler is intentionally more verbose than the console. DEBUG
    records — quartile thresholds, per-month medians and similar
    intermediate values — are therefore preserved in ``pipeline.log`` for
    audit without cluttering an ordinary console run, which is what the
    brief asks for when it requires DEBUG messages to stay out of normal
    runs.

    Parameters
    ----------
    log_file:
        Destination for the file handler. ``None`` disables file logging,
        which is useful for the CLI application and for tests.
    console_level:
        Threshold for console output. Raised to DEBUG when ``debug`` is set.
    file_level:
        Threshold for the log file.
    debug:
        Convenience switch that puts the console into DEBUG mode, used by
        the ``--debug`` flag on the pipeline scripts.
    append:
        ``True`` keeps the accumulated history across runs; ``False``
        truncates the file first.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    if debug:
        console_level = logging.DEBUG

    root = logging.getLogger()
    # Root must sit at the most permissive level in use, otherwise it
    # filters records before either handler ever sees them.
    root.setLevel(min(console_level, file_level))

    # Clear inherited handlers so that repeated calls (for example when a
    # notebook re-runs a cell) do not duplicate every line of output.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            path, mode="a" if append else "w", encoding="utf-8"
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Third-party libraries are noisy at INFO; keep them at WARNING so the
    # project's own trail stays readable.
    for noisy in ("matplotlib", "PIL", "urllib3", "numexpr", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def log_stage_banner(logger: logging.Logger, title: str) -> None:
    """Write a visually distinct stage separator into the log."""
    logger.info("=" * 72)
    logger.info("STAGE: %s", title)
    logger.info("=" * 72)
