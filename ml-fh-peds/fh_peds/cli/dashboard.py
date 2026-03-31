"""Dashboard CLI command."""

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


def main(
    results_dir: Annotated[
        Path,
        typer.Argument(help="Results directory containing experiment outputs."),
    ],
    dashboard_dir: Annotated[
        Path,
        typer.Option(
            help="Path to dashboard directory. Auto-discovered if not provided.",
        ),
    ] = None,
    port: Annotated[int, typer.Option(help="Server port.")] = 8000,
    no_open_browser: Annotated[
        bool, typer.Option(help="Don't auto-open browser.")
    ] = False,
) -> None:
    """Build and serve experiment dashboard.
    
    This command scans the results directory, generates interactive plots
    from experiment data, and serves a static web dashboard.
    
    Example:
        fh dashboard ./results
    """
    # Validate results directory
    if not results_dir.exists():
        typer.echo(f"Error: Results directory not found: {results_dir}", err=True)
        raise typer.Exit(1)

    # Auto-discover dashboard directory
    if dashboard_dir is None:
        # Look for dashboard directory relative to this file
        potential_paths = [
            Path(__file__).parent.parent.parent.parent / "dashboard",  # adjacent to ml-fh-peds
            Path(__file__).parent.parent.parent.parent.parent / "fh-peds-dashboard",  # root level
        ]
        for path in potential_paths:
            if (path / "package.json").exists():
                dashboard_dir = path
                break

        if dashboard_dir is None:
            typer.echo(
                "Error: Dashboard directory not found.\n"
                "Expected one of:\n"
                + "\n".join(f"  - {p}" for p in potential_paths),
                err=True,
            )
            raise typer.Exit(1)

    typer.echo(f"Using dashboard from: {dashboard_dir}")
    typer.echo(f"Loading experiments from: {results_dir}")

    # Check if Node.js is available
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        typer.echo(
            "Error: npm not found. Please install Node.js to use the dashboard.\n"
            "Visit: https://nodejs.org/",
            err=True,
        )
        raise typer.Exit(1)

    # Install dependencies if needed
    node_modules = dashboard_dir / "node_modules"
    if not node_modules.exists():
        typer.echo("Installing dashboard dependencies...")
        result = subprocess.run(
            ["npm", "install"],
            cwd=str(dashboard_dir),
        )
        if result.returncode != 0:
            typer.echo("Error: Failed to install dependencies", err=True)
            raise typer.Exit(1)

    # Run builder
    typer.echo("Building dashboard...")
    builder_args = [
        sys.executable,
        "-m",
        "builder",
        "--results-dir",
        str(results_dir),
        "--dashboard-dir",
        str(dashboard_dir),
        "--port",
        str(port),
    ]

    if no_open_browser:
        builder_args.append("--no-open-browser")

    result = subprocess.run(builder_args, cwd=str(dashboard_dir))
    raise typer.Exit(result.returncode)


if __name__ == "__main__":
    main()
