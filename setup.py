"""Entry point compatible with the official isolated, embeddable Python runtime."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wuwa_mfg.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
