from typing import Union, Tuple

import numpy as np

from .profile import Profile
from pyroll.core import Transport, Unit, Hook
from .constants import RADIATION_COEFFICIENT
import scipy.optimize as scopt


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

    increments = np.zeros_like(p.ring_temperatures)

    source_density = 0  # TODO source density term in W / m^3

    cross_section = np.pi * (p.ring_boundaries[1]) ** 2
    increments[0] = unit.duration / (p.density * p.thermal_capacity * cross_section) * (
            np.pi * p.thermal_conductivity * (p.ring_temperatures[1] - p.ring_temperatures[0])
            + source_density * cross_section
    )

    cross_section = np.pi * (p.ring_boundaries[-1] ** 2 - p.ring_boundaries[-2] ** 2)
    increments[-1] = unit.duration / (p.density * p.thermal_capacity * cross_section) * (
            2 * np.pi
            * (
                    (
                            transport.heat_transfer_factor
                            * (transport.environment_temperature - p.surface_temperature)
                            + RADIATION_COEFFICIENT * transport.relative_radiation_coefficient
                            * (transport.environment_temperature ** 4 - p.surface_temperature ** 4)
                    )
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


@Transport.OutProfile.ring_temperatures
def ring_temperatures_one_step(self: Union[Transport.OutProfile, Profile]):
    if self.transport().disk_element_count == 0:
        transport = self.transport()

        increments = get_increments(transport, transport)

        return transport.in_profile.ring_temperatures[1] + increments


@Transport.DiskElement.OutProfile.ring_temperatures
def ring_temperatures_disk(self: Union[Transport.OutProfile, Profile]):
    transport = self.transport()
    disk = self.unit()

    increments = get_increments(disk, transport)

    return disk.in_profile.ring_temperatures + increments


def _surface_temperature(self: Union[Transport.Profile, Profile]):
    transport: TransportExt = self.transport()

    def f(ts):
        return (
                transport.heat_transfer_factor
                * (transport.environment_temperature - ts)
                + RADIATION_COEFFICIENT * transport.relative_radiation_coefficient
                * (transport.environment_temperature ** 4 - ts ** 4)
                - 2 * self.thermal_conductivity
                * (ts - self.ring_temperatures[-1])
                / (self.equivalent_radius - self.rings[-1])
        )

    def fprime(ts):
        return (
                -4 * RADIATION_COEFFICIENT * transport.relative_radiation_coefficient
                * ts ** 3
                - transport.heat_transfer_factor
                - 2 * self.thermal_conductivity / (self.equivalent_radius - self.rings[-1])
        )

    sol = scopt.root_scalar(f=f, fprime=fprime, x0=self.ring_temperatures[-1], method="newton")

    if sol.converged:
        return sol.root


@Transport.Profile.surface_temperature
def surface_temperature(self: Union[Transport.Profile, Profile]):
    return _surface_temperature(self)


@Transport.DiskElement.Profile.surface_temperature
def disk_surface_temperature(self: Union[Transport.Profile, Profile]):
    return _surface_temperature(self)
