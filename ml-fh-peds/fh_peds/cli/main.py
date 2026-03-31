import typer

from fh_peds.cli import dashboard, train


app = typer.Typer(add_completion=False)
app.command(name="train", short_help="Train FH-PEDS model.")(train.main)
app.command(name="dashboard", short_help="View experiment results dashboard.")(
    dashboard.main
)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
