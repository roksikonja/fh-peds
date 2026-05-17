import typer

from fh_peds.cli import train

app = typer.Typer(add_completion=False)
app.command(name="train", short_help="Train FH-PEDS model.")(train.main)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
