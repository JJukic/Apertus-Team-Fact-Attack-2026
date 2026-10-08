import sys

from src.cli import app

if __name__ == "__main__":
    # Official contract: the container is called with `--input <cases> --output <predictions>` and no command
    if len(sys.argv) > 1 and sys.argv[1] in ("--input", "-i"):
        sys.argv.insert(1, "run")
    app()
