import time

# Begin before importing the CLI and its inference dependencies. The official
# proxy remains authoritative for interpreter/container startup and shutdown.
_invocation_started = time.perf_counter()

import os
import sys

# Official containers pass options directly, without the development 'run'
# subcommand. Enable the JSONL contract before importing CLI configuration.
if len(sys.argv) > 1 and sys.argv[1].startswith("-") and any(
        flag in sys.argv for flag in ("--input", "-i", "--output", "-o")):
    sys.argv.insert(1, "run")
    os.environ["NLI_OFFICIAL_IO"] = "true"

import src.cli as cli

if __name__ == "__main__":
    cli._entrypoint_started = _invocation_started
    cli.app()
