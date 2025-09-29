import logging
import webbrowser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyroll.core as pr
import pyroll.ring_model_thermal
from pathlib import Path

hwang_data_dir = Path(__file__).parent / 'hwang_data'

@pr.RollPass.DiskElement.strain_rate
def strain_rate(self: pr.RollPass.DiskElement):
    return self.roll_pass.strain_rate


def yield_data_from_profiles(u: pr.Unit, name):
    if isinstance(u, pr.PassSequence):
        for su in u.subunits:
            if isinstance(su, (pr.TwoRollPass, pr.Transport, pr.CoolingPipe)):
                if su.subunits:
                    for ssu in su.subunits:
                        yield getattr(ssu.in_profile, name, None)
                        yield getattr(ssu.out_profile, name, None)

def test_solve_hwang(tmp_path: Path, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="pyroll")

    from pyroll.ring_model import Config

    monkeypatch.setattr(Config, "RING_COUNT", 20)
    in_profile = pr.Profile.round(
        radius=25e-3,
        material=["C20", "steel"],
        temperature=1150 + 273.15,
        density=7.5e3,
        specific_heat_capacity=690,
        thermal_conductivity=23,
        flow_stress=100e6,
        relative_radiation_coefficient=0.7
    )

    in_profile.ring_temperatures = in_profile.ring_temperatures

    sequence = pr.PassSequence([
        pr.Transport(
            duration=65,
            heat_transfer_coefficient=30,
            disk_element_count=50
        ),
        pr.RollPass(
            label="Oval",
            orientation="v",
            roll=pr.Roll(
                groove=pr.CircularOvalGroove(
                    depth=15.5e-3,
                    r1=1e-3,
                    r2=46e-3,
                ),
                nominal_radius=200e-3,
                rotational_frequency=0.16,
                temperature=21 + 273.15,
                heat_transfer_coefficient=24000
            ),
            gap=5e-3,
            disk_element_count=50
        ),
        pr.Transport(
            duration=15,
            heat_transfer_coefficient=30,
            disk_element_count=50
        ),
    ])

    sequence.solve(in_profile)

    time = np.array(list(yield_data_from_profiles(sequence, "t")))
    core_temperature = np.array(list(yield_data_from_profiles(sequence, "core_temperature"))) - 273.15
    surface_temperature = np.array(list(yield_data_from_profiles(sequence, "surface_temperature"))) - 273.15

    valid_mask = np.diff(time) >= 0

    time = time[1:][valid_mask]
    core_temperature = core_temperature[1:][valid_mask]
    surface_temperature = surface_temperature[1:][valid_mask]

    surface_temperature_measured = pd.read_csv(hwang_data_dir / "surface_temperature_hwang.csv").surface_temperature
    surface_temperature_time = pd.read_csv(hwang_data_dir / "surface_temperature_hwang.csv").time
    core_temperature_measured = pd.read_csv(hwang_data_dir / "core_temperature_hwang.csv").core_temperature
    core_temperature_time = pd.read_csv(hwang_data_dir / "core_temperature_hwang.csv").time

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid()
    ax.set_ylim(500, 1100)
    ax.set_xlim(50, 80)
    ax.set_ylabel("Temperatures [°C]")
    ax.set_xlabel("Process Time [s]")
    ax.plot(time, surface_temperature, label="Surface Temperature (Calculated)")
    ax.plot(surface_temperature_time, surface_temperature_measured, color="C0", ls="--",
            label="Surface Temperature (Hwang et al.)")
    ax.plot(time, core_temperature, label="Core Temperature (Calculated)")
    ax.plot(core_temperature_time, core_temperature_measured, color="C1", ls="--",
            label="Core Temperature (Hwang et al.)")
    ax.legend()
    fig.show()