"""Shared pytest configuration for the MUISCA test suite.

Adds the repository root to ``sys.path`` so tests can import ``utils``, ``models`` and
``scripts`` directly, and registers the ``data`` marker for tests that need the real
MURaM / MODEST / NICOLE files on the cluster. Those tests skip themselves when the files
are missing, so the synthetic tests run anywhere in well under a minute.

Run everything with ``pytest tests/``; only the synthetic ones with ``pytest -m "not data" tests/``.
"""
import os
import sys
from pathlib import Path

# Never open plot windows during tests (several analysis helpers call plt.show()).
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    """Register the ``data`` marker so ``pytest -m "not data"`` can deselect tests that need real files."""
    config.addinivalue_line(
        "markers", "data: needs the real simulation/observation files (skipped when absent)"
    )
