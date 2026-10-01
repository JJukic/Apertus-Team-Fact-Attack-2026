"""Hack Apertus Track 2A (OST) Source Package."""

import sys
from pathlib import Path

# Ensure track_2a is in sys.path so 'src' imports work from any working directory
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))
