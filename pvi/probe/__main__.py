"""Entry point: `python -m pvi.probe`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
