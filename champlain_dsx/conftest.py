"""Make the ``champlain_dsx`` package importable when running pytest from here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
