"""Make the src-layout package importable for local stdlib unittest runs."""
from __future__ import annotations

import sys
from pathlib import Path

src = Path(__file__).resolve().parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
