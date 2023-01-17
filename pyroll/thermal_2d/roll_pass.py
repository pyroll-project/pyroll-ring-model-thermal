from typing import Union, Tuple

import numpy as np

from .profile import Profile
from pyroll.core import RollPass, Unit, Hook


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

    increments = np.zeros_like(p.ring_temperatures)

    source_density = roll_pass.deformation_heat_efficiency * p.flow_stress * roll_pass.strain_rate

    cross_section = np.pi * (p.ring_boundaries[1]) ** 2
    increments[0] = unit.duration / (p.density * p.thermal_capacity * cross_section) * (
            np.pi * p.thermal_conductivity * (p.ring_temperatures[1] - p.ring_temperatures[0])
            + source_density * cross_section
    )

    cross_section = np.pi * (p.ring_boundaries[-1] ** 2 - p.ring_boundaries[-2] ** 2)
    increments[-1] = unit.duration / (p.density * p.thermal_capacity * cross_section) * (
            2 * np.pi
            * (
                    roll_pass.heat_transfer_factor * (roll_pass.roll.temperature - p.surface_temperature)
                    * p.ring_boundaries[-1]
                    - p.thermal_conductivity * (p.ring_temperatures[-1] - p.ring_temperatures[-2])
                    / (p.rings[-1] - p.rings[-2])
                    * p.ring_boundaries[-2]
            )
            + source_density * cross_section
    )

    for i in range(1, len(increments) - 1):
        cross_section = np.pi * (p.ring_boundaries[i + 1] ** 2 - p.ring_boundaries[i] ** 2)
        increments[i] = unit.duration / (p.density * p.thermal_capacity * cross_section) * (
                2 * np.pi * p.thermal_conductivity
                * (
                        (p.ring_temperatures[i + 1] - p.ring_temperatures[i]) * p.ring_boundaries[i + 1]
                        / (p.rings[i + 1] - p.rings[i])
                        - (p.ring_temperatures[i] - p.ring_temperatures[i - 1]) * p.ring_boundaries[i]
                        / (p.rings[i] - p.rings[i - 1])
                )
                + source_density * cross_section
        )

    return increments


@RollPass.OutProfile.ring_temperatures
def ring_temperatures_one_step(self: Union[RollPass.OutProfile, Profile]):
    if self.roll_pass().disk_element_count == 0:
        roll_pass = self.roll_pass()

        increments = get_increments(roll_pass, roll_pass)

        return roll_pass.in_profile.ring_temperatures + increments


@RollPass.DiskElement.OutProfile.ring_temperatures
def ring_temperatures_disk(self: Union[RollPass.OutProfile, Profile]):
    roll_pass = self.roll_pass()
    disk = self.unit()

    increments = get_increments(disk, roll_pass)

    return disk.in_profile.ring_temperatures + increments
