import logging
import webbrowser
from pathlib import Path

from pyroll.core import Profile, Roll, RollPass, Transport, RoundGroove, CircularOvalGroove, PassSequence


def test_solve(tmp_path: Path, caplog):
    caplog.set_level(logging.INFO, logger="pyroll")

    import pyroll.ring_model_thermal

    in_profile = Profile.round(
        diameter=30e-3,
        temperature=1200 + 273.15,
        strain=0,
        material=["C45", "steel"],
        flow_stress=100e6,
        density=7.5e3,
        thermal_capacity=690,
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
                    rotational_frequency=1,
                    temperature=293,
                ),
                gap=2e-3,
                disk_element_count=20,
            ),
            Transport(
                label="I => II",
                duration=2,
                environment_temperature=293,
                disk_element_count=20,
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
            Transport(
                label="cooling",
                duration=40,
                environment_temperature=293,
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
        f.write_text(result)
        webbrowser.open(f.as_uri())

    except ImportError:
        pass
