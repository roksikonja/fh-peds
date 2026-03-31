"""
Training entry point — thin shim over ``fh_peds.train``.

Run directly:
    python training.py [options]

Or via the installed CLI command:
    fh-peds train [options]
"""

from fh_peds.train import main

if __name__ == "__main__":
    main()
