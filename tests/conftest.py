"""
Test-session guard for the compiled CUDA extension.

`setup.py build_ext --inplace` is not re-run automatically when kernel sources
change, so a stale node_kernel*.so silently exercises old kernels. Fail fast
with rebuild instructions instead of measuring the wrong binary.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT / "warpgbm" / "cuda"
SOURCE_PATTERNS = ("*.cu", "*.cpp", "*.cuh", "*.h")
REBUILD_HINT = "Build/rebuild it: .venv/bin/python setup.py build_ext --inplace"


def pytest_configure(config):
    sources = []
    for pattern in SOURCE_PATTERNS:
        sources.extend(CUDA_DIR.glob(pattern))
    if not sources:
        return  # no in-tree kernel sources to compare against

    try:
        from warpgbm.cuda import node_kernel
    except Exception:
        return  # let test collection surface the real import error

    binary = Path(node_kernel.__file__)
    newest_source = max(s.stat().st_mtime for s in sources)
    if binary.stat().st_mtime < newest_source:
        stale = sorted(s.name for s in sources if s.stat().st_mtime > binary.stat().st_mtime)
        pytest.exit(
            "WarpGBM CUDA extension is older than its kernel sources: "
            f"{', '.join(stale)} changed after the last build. "
            f"Tests would silently exercise stale kernels.\n{REBUILD_HINT}",
            returncode=4,
        )
