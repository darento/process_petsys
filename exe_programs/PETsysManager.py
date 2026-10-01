#!/usr/bin/env python3
"""PETsys Manager — Cornell acquisition, conversion, calibration, LM and QC workflows.

Run from the process_petsys environment on the Cornell Linux machine:
    python exe_programs/PETsysManager.py [--profile PATH]

A separate application from LDAT Inspector (exe_programs/LDATInspector.py).
"""

from multiprocessing import freeze_support
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __package__:
    from .petsys_manager_gui import PETsysManager, __version__, main
else:
    from petsys_manager_gui import PETsysManager, __version__, main


if __name__ == "__main__":
    freeze_support()
    main()
