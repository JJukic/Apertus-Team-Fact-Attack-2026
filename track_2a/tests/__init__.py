# Hack Apertus Track 2A Test Suite
import sys
from pathlib import Path

_track2a_dir = str(Path(__file__).resolve().parent.parent)
if _track2a_dir not in sys.path:
    sys.path.insert(0, _track2a_dir)
