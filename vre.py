#!/usr/bin/env python
"""Entry point so the CLI runs without installing the package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vre.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
