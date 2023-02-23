from typing import Union

import numpy as np
import scipy.optimize as scopt
from pyroll.core import Transport, Unit, Hook

from .constants import RADIATION_COEFFICIENT
from .profile import Profile


@Transport.extension_class
class TransportExt(Transport):
    heat_transfer_coefficient = Hook[float]()
    """Heat transfer coefficient by convection to atmosphere."""


@TransportExt.heat_transfer_coefficient
def heat_transfer_coefficient(self: Transport):
    return 15


def get_increments(unit: Unit, transport: TransportExt, ring_temperatures) -> np.ndarray:
    p: Profile = unit.out_profile

    increments = np.zeros_like(ring_temperatures)

    source_density = 0  # TODO source density term in W / m^3

    cross_section = p.ring_sections[0].area
    increments[0] = unit.duration / (p.density * p.thermal_capacity * cross_section) * (
            (ring_temperatures[1] - ring_temperatures[0]) * p.ring_contours[1].length
            / p.rings[1] * p.thermal_conductivity
            + source_density * cross_section
    )

    cross_section = p.ring_sections[-1].area
    increments[-1] = unit.duration / (p.density * p.thermal_capacity * cross_section) * (
            (
                    transport.heat_transfer_coefficient
                    * (transport.environment_temperature - p.surface_temperature)
                    + RADIATION_COEFFICIENT * p.relative_radiation_coefficient
                    * (transport.environment_temperature ** 4 - p.surface_temperature ** 4)
            )
            * p.ring_contours[-1].length
            - p.thermal_conductivity * (ring_temperatures[-1] - ring_temperatures[-2])
            / (p.rings[-1] - p.rings[-2])
            * p.ring_contours[-2].length
            + source_density * cross_section
    )

    for i in range(1, len(increments) - 1):
        cross_section = p.ring_sections[i].area
        increments[i] = unit.duration / (p.density * p.thermal_capacity * cross_section) * (
                p.thermal_conductivity
                * (
                        (ring_temperatures[i + 1] - ring_temperatures[i]) * p.ring_contours[i + 1].length
                        / (p.rings[i + 1] - p.rings[i])
                        - (ring_temperatures[i] - ring_temperatures[i - 1]) * p.ring_contours[i].length
                        / (p.rings[i] - p.rings[i - 1])
                )
                + source_density * cross_section
        )

    return increments


@Transport.DiskElement.OutProfile.ring_temperatures
def ring_temperatures_disk(self: Union[Transport.DiskElement.OutProfile, Profile]):
    transport = self.transport
    disk = self.disk_element

    x0 = get_increments(disk, transport, disk.out_profile.ring_temperatures)

    def f(x):
        out_ring_temperatures = disk.in_profile.ring_temperatures + x
        return get_increments(disk, transport, out_ring_temperatures) - x

    sol = scopt.root(f, x0=x0)

    if not sol.success:
        raise RuntimeError(f"Numerical procedure did not succeed: {sol.message}.")

    return disk.in_profile.ring_temperatures + sol.x


def _surface_temperature(self: Union[Transport.Profile, Profile]):
    transport: TransportExt = self.transport

    def f(ts):
        return (
                transport.heat_transfer_coefficient
                * (transport.environment_temperature - ts)
                + RADIATION_COEFFICIENT * self.relative_radiation_coefficient
                * (transport.environment_temperature ** 4 - ts ** 4)
                - self.thermal_conductivity
                * (ts - self.ring_temperatures[-1])
                / (self.equivalent_radius - self.rings[-1])
        )

    def fprime(ts):
        return (
                -4 * RADIATION_COEFFICIENT * self.relative_radiation_coefficient
                * ts ** 3
                - transport.heat_transfer_coefficient
                - self.thermal_conductivity / (self.equivalent_radius - self.rings[-1])
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
