#!/usr/bin/env python3
"""PETsys LDAT Inspector — offline multi-file workbench.

Run from the project root in the process_petsys environment:
    python exe_programs/LDATInspector.py

The original Tk implementation is preserved in LDATInspector_legacy.py.
"""

from multiprocessing import freeze_support
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __package__:
    from .ldat_inspector_gui import LDATWorkbench, __version__, main
else:
    from ldat_inspector_gui import LDATWorkbench, __version__, main


if __name__ == "__main__":
    freeze_support()
    main()
