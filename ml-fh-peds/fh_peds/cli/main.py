import typer

from fh_peds.cli import train

app = typer.Typer()
app.command("train")(train.main)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
