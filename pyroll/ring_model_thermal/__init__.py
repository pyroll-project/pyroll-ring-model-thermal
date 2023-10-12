VERSION = "2.1.1"

from . import profile
from . import roll_pass
from . import transport
from .config import Config

from pyroll.core import root_hooks, Unit

root_hooks.add(Unit.OutProfile.temperature)
root_hooks.add(Unit.OutProfile.ring_temperatures)

import importlib.util

REPORT_INSTALLED = bool(importlib.util.find_spec("pyroll.report"))

if REPORT_INSTALLED:
    from . import report
    import pyroll.report

    pyroll.report.plugin_manager.register(report)
