import logging
import webbrowser
import numpy as np
import scipy.interpolate as interpolate

from pathlib import Path
from pyroll.core import Profile, Roll, RollPass, Transport, RoundGroove, CircularOvalGroove, PassSequence
import pyroll.ring_model_thermal

demo_htc = np.linspace(1, 200, 20)
demo_lengths = np.linspace(0, 10, 20)
htc_fun = interpolate.interp1d(demo_lengths, demo_htc)


@RollPass.DiskElement.strain_rate
def strain_rate(self: RollPass.DiskElement):
    return self.roll_pass.strain_rate


@Transport.DiskElement.heat_transfer_coefficient
def heat_transfer_coefficient(self: Transport.DiskElement):
    def f(position):
        return htc_fun(self.in_profile.x)
    return f


def test_solve(tmp_path: Path, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="pyroll")
    monkeypatch.setenv("PYROLL_REPORT_PRINT_DISK_ELEMENTS", "True")

    from pyroll.ring_model import Config

    monkeypatch.setattr(Config, "RING_COUNT", 20)

    in_profile = Profile.round(
        diameter=30e-3,
        temperature=1200 + 273.15,
        strain=0,
        material=["C45", "steel"],
        flow_stress=100e6,
        density=7.5e3,
        specific_heat_capacity=690,
        thermal_conductivity=28,
    )

    sequence = PassSequence(
        [
            RollPass(
                label="Oval I",
                roll=Roll(
                    groove=CircularOvalGroove(
                        depth=8e-3,
                        r1=6e-3,
                        r2=40e-3
                    ),
                    nominal_radius=160e-3,
                    temperature=293,
                ),
                gap=2e-3,
                disk_element_count=20,
                velocity=0.23
            ),
            Transport(
                label="I => II",
                length=0.5,
                environment_temperature=293,
                disk_element_count=20
            ),
            RollPass(
                label="Round II",
                roll=Roll(
                    groove=RoundGroove(
                        r1=1e-3,
                        r2=12.5e-3,
                        depth=11.5e-3
                    ),
                    nominal_radius=160e-3,
                    rotational_frequency=1,
                    temperature=293,
                ),
                gap=2e-3,
                disk_element_count=20,
            ),
        ]
    )

    try:
        sequence.solve(in_profile)
    finally:
        print("\nLog:")
        print(caplog.text)

    try:
        import pyroll.report
        result = pyroll.report.report(sequence)
        f = (tmp_path / "report.html")
        f.write_text(result, encoding="utf-8")
        webbrowser.open(f.as_uri())

    except ImportError:
        pass
