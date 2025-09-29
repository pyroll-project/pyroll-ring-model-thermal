import logging
import webbrowser
import numpy as np


from pathlib import Path

from matplotlib import pyplot as plt
from pyroll.core import root_hooks, Unit, DiskElementUnit, Profile, Roll, RollPass, Transport, FalseRoundGroove, CircularOvalGroove, PassSequence, Hook
import pyroll.ring_model_thermal


Profile.global_position = Hook[float]()
"""Global Position of the Profile in the rolling train."""

@Unit.OutProfile.global_position
def out_global_position(self: Unit.OutProfile):
    return self.unit.in_profile.global_position + self.unit.length


@DiskElementUnit.DiskElement.InProfile.global_position
def in_x(self: DiskElementUnit.DiskElement.InProfile):
    try:
        return self.disk_element.prev.out_profile.global_position
    except IndexError:
        return self.disk_element.parent.in_profile.global_position


root_hooks.append(Unit.OutProfile.global_position)


@RollPass.DiskElement.strain_rate
def strain_rate(self: RollPass.DiskElement):
    return self.roll_pass.strain_rate


def yield_data_from_profiles(u: Unit, name):
    if isinstance(u, PassSequence):
        for su in u.subunits:
            if isinstance(su, (RollPass, Transport)):
                if su.subunits:
                    for ssu in su.subunits:
                        yield getattr(ssu.in_profile, name, None)
                        yield getattr(ssu.out_profile, name, None)

def test_solve(tmp_path: Path, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="pyroll")

    from pyroll.ring_model import Config

    monkeypatch.setattr(Config, "RING_COUNT", 20)

    in_profile = Profile.round(
        diameter=17e-3,
        temperature=1050 + 273.15,
        strain=0,
        material=["BST500", "steel"],
        density=7.5e3,
        specific_heat_capacity=690,
        thermal_conductivity=23,
        flow_stress=100e6,
        global_position=0
    )

    sequence = PassSequence([RollPass(
        label="1",
        orientation="h",
        roll=Roll(
            groove=CircularOvalGroove(
                depth=4.35e-3,
                r1=1.8e-3,
                r2=18.50e-3,
            ),
            nominal_radius=208e-3 / 2,
            rotational_frequency=1236 / 60,
            temperature=293,
        ),
        gap=1.8e-3,
        coulomb_friction_coefficient=0.4,
        disk_element_count=20,
    ),
        Transport(
            label="1 => 2",
            length=0.85
        ),
        RollPass(
            label="2",
            orientation="v",
            roll=Roll(
                groove=FalseRoundGroove(
                    depth=6e-3,
                    r1=1.550e-3,
                    r2=7.1e-3,
                    flank_angle=60
                ),
                nominal_radius=208e-3 / 2,
                rotational_frequency=1545 / 60,
                temperature=293,
            ),
            gap=1.55e-3,
            coulomb_friction_coefficient=0.4,
            disk_element_count=20
        ),
        Transport(
            label="2 => 3",
            length=0.85
        ),
        RollPass(
            label="3",
            orientation="h",
            roll=Roll(
                groove=CircularOvalGroove(
                    depth=3.6e-3,
                    r1=1.4e-3,
                    r2=15e-3
                ),
                nominal_radius=208e-3 / 2,
                rotational_frequency=1934 / 60,
                temperature=293,
            ),
            gap=1.4e-3,
            coulomb_friction_coefficient=0.4,
            disk_element_count=20
        ),
        Transport(
            label="3 => 4",
            length=0.85
        ),
        RollPass(
            label="4",
            orientation="v",
            roll=Roll(
                groove=FalseRoundGroove(
                    depth=4.75e-3,
                    r1=1.5e-3,
                    r2=5.61e-3,
                    flank_angle=65
                ),
                nominal_radius=208e-3 / 2,
                rotational_frequency=2417 / 60,
                temperature=293,
            ),
            gap=1.5e-3,
            coulomb_friction_coefficient=0.4,
            disk_element_count=20
        ),
        Transport(
            label="4 => 5",
            length=0.85
        ),
        RollPass(
            label="5",
            orientation="h",
            roll=Roll(
                groove=CircularOvalGroove(
                    depth=3e-3,
                    r1=1.1e-3,
                    r2=13e-3
                ),
                nominal_radius=208e-3 / 2,
                rotational_frequency=3022 / 60,
                temperature=293,
            ),
            gap=1.45e-3,
            coulomb_friction_coefficient=0.4,
            disk_element_count=20
        ),
        Transport(
            label="5 => 6",
            length=0.85
        ),
        RollPass(
            label="6",
            orientation="v",
            roll=Roll(
                groove=FalseRoundGroove(
                    depth=3.75e-3,
                    r1=1.5e-3,
                    r2=4.59e-3,
                    flank_angle=65
                ),
                nominal_radius=158.8e-3 / 2,
                rotational_frequency=5024 / 60,
                temperature=293,
            ),
            gap=1.5e-3,
            coulomb_friction_coefficient=0.4,
            disk_element_count=20
        ),
        Transport(
            label="6 => 7",
            length=0.85
        ),
        RollPass(
            label="7",
            orientation="h",
            roll=Roll(
                groove=CircularOvalGroove(
                    depth=2e-3,
                    r1=1.4e-3,
                    r2=10.2e-3
                ),
                nominal_radius=158.8e-3 / 2,
                rotational_frequency=6201 / 60,
                temperature=293,
            ),
            gap=1.4e-3,
            coulomb_friction_coefficient=0.4,
            disk_element_count=20
        ),
        Transport(
            label="7 => 8",
            length=0.85
        ),
        RollPass(
            label="8",
            orientation="v",
            roll=Roll(
                groove=FalseRoundGroove(
                    depth=2.75e-3,
                    r1=1.2e-3,
                    r2=3.57e-3,
                    flank_angle=65
                ),
                nominal_radius=158.8e-3 / 2,
                rotational_frequency=7849 / 60,
                temperature=293,
            ),
            gap=1.2e-3,
            coulomb_friction_coefficient=0.4,
            disk_element_count=20
        ),
        Transport(
            label="8 => 9",
            length=0.85
        ),
        RollPass(
            label="9",
            orientation="h",
            roll=Roll(
                groove=CircularOvalGroove(
                    depth=1.5e-3,
                    r1=1e-3,
                    r2=8.5e-3
                ),
                nominal_radius=158.8e-3 / 2,
                rotational_frequency=9824 / 60,
                temperature=293,
            ),
            gap=1.25e-3,
            coulomb_friction_coefficient=0.4,
            disk_element_count=20
        ),
        Transport(
            label="9 => 10",
            length=0.85
        ),
        RollPass(
            label="10",
            orientation="v",
            roll=Roll(
                groove=FalseRoundGroove(
                    depth=2.25e-3,
                    r1=1e-3,
                    r2=2.810e-3,
                    flank_angle=60
                ),
                nominal_radius=158.8e-3 / 2,
                rotational_frequency=12281 / 60,
                temperature=293,
            ),
            gap=1e-3,
            coulomb_friction_coefficient=0.4,
            disk_element_count=20
        )

    ])

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

    global_position = np.array(list(yield_data_from_profiles(sequence, "global_position")))
    core_temperature = np.array(list(yield_data_from_profiles(sequence, "core_temperature")))
    mean_temperature = np.array(list(yield_data_from_profiles(sequence, "temperature")))
    surface_temperature = np.array(list(yield_data_from_profiles(sequence, "surface_temperature")))

    fig, ax = plt.subplots()
    ax.set_title("Temperatures over Position")
    ax.set_xlabel("Global Position [m]")
    ax.set_ylabel("Temperature [°C]")
    ax.grid(True)
    ax.plot(global_position, core_temperature, color='C0', label="Core Temperature")
    ax.plot(global_position, mean_temperature, color='C1', label="Mean Temperature")
    ax.plot(global_position, surface_temperature, color='C2', label="Surface Temperature")
    ax.legend()
    fig.show()
