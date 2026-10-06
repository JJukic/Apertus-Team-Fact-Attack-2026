"""Load the hash-verified, offline-only client used in the October 5 baseline."""
import hashlib
import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "docs/evaluation/frozen_sources/apertus_client_c0bd757.py"
SOURCE_SHA256 = "78b574a4b0deb8693951cb1ef240c9d3053a148c259ddb84c15235af7056d80c"


def load_client(expected_sha256=SOURCE_SHA256):
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Archived Apertus client differs from the recorded baseline")
    spec = importlib.util.spec_from_file_location("historical_apertus", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ApertusClient


ApertusClient = load_client()
