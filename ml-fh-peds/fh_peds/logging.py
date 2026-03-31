import logging
from datetime import datetime
from pathlib import Path


def setup_run(base_dir: Path) -> tuple[Path, logging.Logger]:
    """Create a timestamped results directory and configure logging to stdout and ``stdout.log``."""
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
