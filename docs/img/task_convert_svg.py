import subprocess
from typing import Any

import pytask
from pathlib import Path

THIS_DIR = Path(__file__).parent

for f in THIS_DIR.glob("*.svg"):
    @pytask.mark.task(id=f.name)
    @pytask.mark.depends_on(f)
    @pytask.mark.produces([f.with_suffix(s) for s in [".png", ".pdf"]])
    def task_convert_svg(depends_on: Path, produces: dict[Any, Path]):
        for of in produces.values():
            result = subprocess.run([
                "inkscape",
                "-d", "600",
                "-D",
                "-o", str(of),
                str(depends_on)
            ], text=True)

            result.check_returncode()
