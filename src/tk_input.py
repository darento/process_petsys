"""Tk without an X input method on Linux (bug-tk-ibus-destroy-hang).

With ibus running as the X input method (``XMODIFIERS=@im=ibus``, the Cornell desktop), Tk 8.6.13
intermittently never returns from destroying a text widget, so a window or a test hangs on close.
Tk reads ``XMODIFIERS`` when it opens a root, so this runs before the first root. Plain typing is
unaffected; only ibus text composition is lost. Kept free of heavy imports: ``tests/conftest.py``
calls it before numpy loads.
"""

import os
import sys

NO_INPUT_METHOD = "@im=none"


def disable_x_input_method():
    if sys.platform.startswith("linux"):
        os.environ["XMODIFIERS"] = NO_INPUT_METHOD
