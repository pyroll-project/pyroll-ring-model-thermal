import numpy as np
import scipy.optimize as scopt

from typing import Union
from .config import Config
from .profile import Profile
from pyroll.core import CoolingPipe, Unit, Hook, root_hooks


@CoolingPipe.extension_class
class CoolingPipeExt(CoolingPipe):
    heat_transfer_coefficient = Hook[float]()
    """Heat transfer coefficient by cooling water."""


@CoolingPipe.DiskElement.extension_class
class CoolingPipeExt(CoolingPipe.DiskElement):
    heat_transfer_coefficient = Hook[float]()
    """Heat transfer coefficient by convection to atmosphere."""


@CoolingPipeExt.heat_transfer_coefficient
def heat_transfer_coefficient(self: CoolingPipe):
    """Default value from measurements by H. Wehage; (Beitrag zur rechnergestützten Erarbeitung von Projekten und
    Technologien für kontinuierliche Feinstahl- und Drahtstraßen, PhD, TU Freiberg, 1990)"""
    return 4000


@CoolingPipe.DiskElement.heat_transfer_coefficient
def heat_transfer_coefficient(self: CoolingPipe.DiskElement):
    return self.cooling_pipe.heat_transfer_coefficient


def get_increments(unit: Unit, cooling_pipe: CoolingPipeExt, ring_temperatures) -> np.ndarray:
    p: Profile = unit.out_profile

    increments = np.zeros_like(ring_temperatures)

    cross_section = p.ring_sections[0].area
    increments[0] = unit.duration / (p.density * p.specific_heat_capacity * cross_section) * (
            (ring_temperatures[1] - ring_temperatures[0]) * p.ring_contours[1].length
            / p.rings[1] * p.thermal_conductivity
    )

    cross_section = p.ring_sections[-1].area
    increments[-1] = unit.duration / (p.density * p.specific_heat_capacity * cross_section) * (
            (
                    cooling_pipe.heat_transfer_coefficient
                    * (cooling_pipe.cooling_water_temperature - p.surface_temperature)
            )
            * p.ring_contours[-1].length
            - p.thermal_conductivity * (ring_temperatures[-1] - ring_temperatures[-2])
            / (p.rings[-1] - p.rings[-2])
            * p.ring_contours[-2].length

    )

    for i in range(1, len(increments) - 1):
        cross_section = p.ring_sections[i].area
        increments[i] = unit.duration / (p.density * p.specific_heat_capacity * cross_section) * (
                p.thermal_conductivity
                * (
                        (ring_temperatures[i + 1] - ring_temperatures[i]) * p.ring_contours[i + 1].length
                        / (p.rings[i + 1] - p.rings[i])
                        - (ring_temperatures[i] - ring_temperatures[i - 1]) * p.ring_contours[i].length
                        / (p.rings[i] - p.rings[i - 1])
                )

        )

    return increments


def _solve_step(unit, cooling_pipe, in_ring_temperatures):
    x0 = get_increments(unit, cooling_pipe, in_ring_temperatures)

    def f(x):
        out_ring_temperatures = in_ring_temperatures + x
        return get_increments(unit, cooling_pipe, out_ring_temperatures) - x

    sol = scopt.root(f, x0=x0)

    if not sol.success:
        raise RuntimeError(f"Numerical procedure did not succeed: {sol.message}.")

    return in_ring_temperatures + sol.x


@CoolingPipe.OutProfile.ring_temperatures
def ring_temperatures_disk(self: Union[CoolingPipe.OutProfile, Profile]):
    if not self.cooling_pipe.disk_elements:
        cooling_pipe = self.cooling_pipe

        return _solve_step(cooling_pipe, cooling_pipe, cooling_pipe.in_profile.ring_temperatures)


@CoolingPipe.DiskElement.OutProfile.ring_temperatures
def ring_temperatures_disk(self: Union[CoolingPipe.DiskElement.OutProfile, Profile]):
    cooling_pipe = self.cooling_pipe
    disk = self.disk_element

    return _solve_step(disk, cooling_pipe, disk.in_profile.ring_temperatures)


def _surface_temperature(self: Union[CoolingPipe.Profile, Profile]):
    cooling_pipe: CoolingPipeExt = self.cooling_pipe

    def f(ts):
        return (
                cooling_pipe.heat_transfer_coefficient
                * (cooling_pipe.cooling_water_temperature - ts)
                - self.thermal_conductivity
                * (ts - self.ring_temperatures[-1])
                / (self.equivalent_radius - self.rings[-1])
        )

    def fprime(ts):
        return (

                - cooling_pipe.heat_transfer_coefficient
                - self.thermal_conductivity / (self.equivalent_radius - self.rings[-1])
        )

    sol = scopt.root_scalar(f=f, fprime=fprime, x0=self.ring_temperatures[-1], method="newton")

    if sol.converged:
        return sol.root


@CoolingPipe.Profile.surface_temperature
def surface_temperature(self: Union[CoolingPipe.Profile, Profile]):
    return _surface_temperature(self)


@CoolingPipe.DiskElement.Profile.surface_temperature
def disk_surface_temperature(self: Union[CoolingPipe.Profile, Profile]):
    return _surface_temperature(self)
