VERSION = "2.0.0b"

from .constants import *
from . import profile
from . import roll_pass
from . import transport

from pyroll.core import root_hooks, Unit

root_hooks.add(Unit.OutProfile.temperature)
root_hooks.add(Unit.OutProfile.ring_temperatures)

import importlib.util

REPORT_INSTALLED = bool(importlib.util.find_spec("pyroll.report"))

if REPORT_INSTALLED:
    from . import report
    import pyroll.report
    pyroll.report.plugin_manager.register(report)
