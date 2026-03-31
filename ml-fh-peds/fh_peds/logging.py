"""
Logging and results-directory setup for fh-peds training runs.

Usage
-----
    from fh_peds.logging import setup_run

    results_dir, log = setup_run(base_dir)
    log.info("Training started")
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_run(base_dir: Path) -> tuple[Path, logging.Logger]:
    """Create a timestamped results directory and configure logging to both
    stdout and a ``stdout.log`` file inside that directory.

    Parameters
    ----------
    base_dir:
        Parent directory under which ``results/<timestamp>/`` will be created.

    Returns
    -------
    results_dir:
        The newly created ``results/<timestamp>`` directory.
    log:
        A configured :class:`logging.Logger` for the run.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = base_dir / "results" / timestamp
    results_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(results_dir / "stdout.log"),
        ],
    )
    log = logging.getLogger("fh_peds")
    log.info(f"Results directory: {results_dir}")
    return results_dir, log
