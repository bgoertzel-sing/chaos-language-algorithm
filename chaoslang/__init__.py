"""Local src-layout import shim for uninstalled unittest runs."""
from __future__ import annotations

from pathlib import Path

_src_pkg = Path(__file__).resolve().parents[1] / "src" / "chaoslang"
__path__ = [str(_src_pkg)]
_code = (_src_pkg / "__init__.py").read_text()
exec(compile(_code, str(_src_pkg / "__init__.py"), "exec"), globals())
