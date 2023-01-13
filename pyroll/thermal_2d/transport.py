from typing import Union, Tuple

import numpy as np

from .profile import Profile
from pyroll.core import Transport, Unit, Hook
from .constants import RADIAL_DISCRETIZATION_COUNT, RADIATION_COEFFICIENT


@Transport.extension_class
class TransportExt(Transport):
    heat_transfer_factor = Hook[float]()
    """Heat transfer coefficient by convection to atmosphere."""

    relative_radiation_coefficient = Hook[float]()
    """Heat transfer coefficient by convection to atmosphere."""


@TransportExt.relative_radiation_coefficient
def relative_radiation_coefficient(self: Transport):
    return 0.8


@TransportExt.heat_transfer_factor
def heat_transfer_factor(self: Transport):
    return 15


def get_increments(unit: Unit, transport: TransportExt) -> Tuple[np.ndarray, np.ndarray]:
    p: Profile = unit.in_profile

    dr = p.equivalent_radius / RADIAL_DISCRETIZATION_COUNT
    radii = p.temperature_profile[0]
    temperatures = p.temperature_profile[1]
    increments = np.zeros_like(temperatures)

    source_density = 0  # TODO source density term in W / m^3

    cross_section = np.pi * (dr / 2) ** 2
    increments[0] = unit.duration / (p.density * p.thermal_capacity * cross_section) * (
            np.pi * p.thermal_conductivity * (temperatures[1] - temperatures[0])
            + source_density * cross_section
    )

    cross_section = np.pi * ((radii[-1] + dr / 2) ** 2 - (radii[-1] - dr / 2) ** 2)
    increments[-1] = unit.duration / (p.density * p.thermal_capacity * cross_section) * (
            2 * np.pi
            * (
                    (
                            transport.heat_transfer_factor
                            * (transport.environment_temperature - p.surface_temperature)
                            + RADIATION_COEFFICIENT * transport.relative_radiation_coefficient
                            * (transport.environment_temperature ** 4 - p.surface_temperature ** 4)
                    )
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


@Transport.OutProfile.temperature_profile
def temperature_profile_one_step(self: Union[Transport.OutProfile, Profile]):
    transport = self.transport()

    increments = get_increments(transport, transport)

    return np.array(
        [
            np.linspace(0, self.equivalent_radius, RADIAL_DISCRETIZATION_COUNT),
            transport.in_profile.temperature_profile[1] + increments
        ]
    )


@Transport.DiskElement.duration
def disk_duration(self: Transport.DiskElement):
    return self.transport().duration / self.transport().disk_element_count


@Transport.DiskElement.length
def disk_length(self: Transport.DiskElement):
    return self.transport().length / self.transport().disk_element_count


@Transport.DiskElement.OutProfile.temperature_profile
def temperature_profile_disk(self: Union[Transport.OutProfile, Profile]):
    transport = self.transport()
    disk = self.unit()

    increments = get_increments(disk, transport)

    return np.array(
        [
            np.linspace(0, self.equivalent_radius, RADIAL_DISCRETIZATION_COUNT),
            disk.in_profile.temperature_profile[1] + increments
        ]
    )


@Transport.OutProfile.temperature_profile
def temperature_profile_from_disks(self: Union[Transport.OutProfile, Profile]):
    return self.transport().disk_elements[-1].out_profile.temperature_profile
