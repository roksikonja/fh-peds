"""
Top-level CLI dispatcher for fh-peds.

Usage
-----
    fh-peds train [options]
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fh-peds",
        description="FH pediatric screening model toolkit.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    subparsers.add_parser(
        "train",
        help="Train and evaluate the FH pediatric screening model.",
        add_help=False,  # training module provides its own --help
    )

    args, remaining = parser.parse_known_args()

    if args.command == "train":
        # Rewrite sys.argv so that the training module's own ArgumentParser
        # sees only the training-specific flags (and --help works correctly).
        sys.argv = ["fh-peds train"] + remaining
        from fh_peds.train import main as train_main

        train_main()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
