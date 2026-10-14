"""Convenience entry point for the maintained package; defaults to bounded smoke."""

import sys
from damldg.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["train", "--config", "configs/smoke.json"])
    main()
