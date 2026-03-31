import logging
from pathlib import Path


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to stdout and ``<log_dir>/stdout.log``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "stdout.log"),
        ],
    )
    return logging.getLogger("fh_peds")
