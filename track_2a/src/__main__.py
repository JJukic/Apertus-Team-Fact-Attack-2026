import os
import sys

# Official containers pass options directly, without the development 'run'
# subcommand. Enable the JSONL contract before importing CLI configuration.
if len(sys.argv) > 1 and sys.argv[1].startswith("-") and any(
        flag in sys.argv for flag in ("--input", "-i", "--output", "-o")):
    sys.argv.insert(1, "run")
    os.environ["NLI_OFFICIAL_IO"] = "true"

from src.cli import app

if __name__ == "__main__":
    # Official contract: the container is called with `--input <cases> --output <predictions>` and no command
    if len(sys.argv) > 1 and sys.argv[1] in ("--input", "-i"):
        sys.argv.insert(1, "run")
    app()
