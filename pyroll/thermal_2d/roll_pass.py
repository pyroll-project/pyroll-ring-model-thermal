from typing import Union, Tuple

import numpy as np

from .profile import Profile
from pyroll.core import RollPass, Unit, Hook
from .constants import RADIAL_DISCRETIZATION_COUNT, RADIATION_COEFFICIENT


@RollPass.extension_class
class RollPassExt(RollPass):
    heat_transfer_factor = Hook[float]()
    """Heat transfer coefficient by convection to atmosphere."""

    deformation_heat_efficiency = Hook[float]()
    """Efficiency of heat generation by deformation power."""


@RollPassExt.deformation_heat_efficiency
def deformation_heat_efficiency(self: RollPass):
    return 0.95


@RollPassExt.heat_transfer_factor
def heat_transfer_factor(self: RollPass):
    return 6000


def get_increments(unit: Unit, roll_pass: RollPassExt) -> Tuple[np.ndarray, np.ndarray]:
    p: Profile = unit.in_profile

    dr = p.equivalent_radius / RADIAL_DISCRETIZATION_COUNT
    radii = p.temperature_profile[0]
    temperatures = p.temperature_profile[1]
    increments = np.zeros_like(temperatures)

    source_density = roll_pass.deformation_heat_efficiency * p.flow_stress * roll_pass.strain_rate

    cross_section = np.pi * (dr / 2) ** 2
    increments[0] = unit.duration / (p.density * p.thermal_capacity * cross_section) * (
            np.pi * p.thermal_conductivity * (temperatures[1] - temperatures[0])
            + source_density * cross_section
    )

    cross_section = np.pi * ((radii[-1] + dr / 2) ** 2 - (radii[-1] - dr / 2) ** 2)
    increments[-1] = unit.duration / (p.density * p.thermal_capacity * cross_section) * (
            2 * np.pi
            * (
                    roll_pass.heat_transfer_factor * (roll_pass.roll.temperature - p.surface_temperature)
                    * (radii[-1] + dr / 2)
                    - p.thermal_conductivity * (temperatures[-1] - temperatures[-2]) / dr * (radii[-1] - dr / 2)
            )
            + source_density * cross_section
    )

    for i in range(1, len(increments) - 1):
        cross_section = np.pi * ((radii[i] + dr / 2) ** 2 - (radii[i] - dr / 2) ** 2)
        increments[i] = unit.duration / (p.density * p.thermal_capacity * cross_section) * (
                2 * np.pi * p.thermal_conductivity / dr
                * (
                        (temperatures[i + 1] - temperatures[i]) * (radii[i] + dr / 2)
                        - (temperatures[i] - temperatures[i - 1]) * (radii[i] - dr / 2)
                )
                + source_density * cross_section
        )

    return increments


@RollPass.OutProfile.temperature_profile
def temperature_profile_one_step(self: Union[RollPass.OutProfile, Profile]):
    if self.roll_pass().disk_element_count == 0:
        roll_pass = self.roll_pass()

        increments = get_increments(roll_pass, roll_pass)

        return np.array(
            [
                np.linspace(0, self.equivalent_radius, RADIAL_DISCRETIZATION_COUNT),
                roll_pass.in_profile.temperature_profile[1] + increments
            ]
        )


@RollPass.DiskElement.OutProfile.temperature_profile
def temperature_profile_disk(self: Union[RollPass.OutProfile, Profile]):
    roll_pass = self.roll_pass()
    disk = self.unit()

    increments = get_increments(disk, roll_pass)

    return np.array(
        [
            np.linspace(0, self.equivalent_radius, RADIAL_DISCRETIZATION_COUNT),
            disk.in_profile.temperature_profile[1] + increments
        ]
    )
