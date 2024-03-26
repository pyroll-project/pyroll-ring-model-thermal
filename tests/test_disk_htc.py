import logging
import webbrowser
import numpy as np
import scipy.interpolate as interpolate

from pathlib import Path
from pyroll.core import Profile, CoolingPipe, PassSequence
import pyroll.ring_model_thermal


@CoolingPipe.DiskElement.heat_transfer_coefficient
def heat_transfer_coefficient(self: CoolingPipe.DiskElement):
    cp = self.cooling_pipe
    profile_surface_temperature_celsius = self.in_profile.surface_temperature - 273.15
    equivalent_profile_diameter = cp.in_profile.equivalent_radius * 2
    inner_cooling_pipe_diameter = cp.inner_radius * 2
    diameter_ratio = equivalent_profile_diameter / inner_cooling_pipe_diameter
    velocity_ratio = cp.coolant_velocity / cp.in_profile.velocity
    coolant_cubic_meters_per_hour = cp.coolant_volume_flux * 3600

    htc_surface_temperature = 40000 + 6.402 * 1e6 / profile_surface_temperature_celsius
    htc_water_volume_flux_geometry = 0.061 * coolant_cubic_meters_per_hour ** 0.688 * diameter_ratio ** -0.472
    htc_water_velocity_ratio = 1.028 * velocity_ratio ** -0.187

    return htc_surface_temperature * htc_water_volume_flux_geometry * htc_water_velocity_ratio


def test_solve(tmp_path: Path, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="pyroll")

    from pyroll.ring_model import Config

    monkeypatch.setattr(Config, "RING_COUNT", 20)

    in_profile = Profile.round(
        diameter=30e-3,
        surface_temperature=1200 + 273.15,
        core_temperature=1200 + 273.15,
        strain=0,
        material=["C45", "steel"],
        flow_stress=100e6,
        density=7.5e3,
        specific_heat_capacity=690,
        thermal_conductivity=28,
        velocity=1
    )

    sequence = PassSequence(
        [
            CoolingPipe(
                disk_element_count=30,
                length=1,
                inner_radius=35e-3,
                coolant_volume_flux=0.0125,
                coolant_temperature=35 + 273.15
            )
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
