"""Single entry point: ``python -m block3 <predict|calibrate|probes> ...``."""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from . import probes, runner


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "probes":
        probes.main(args[1:])
        return
    runner.main(args)


if __name__ == "__main__":
    main()
