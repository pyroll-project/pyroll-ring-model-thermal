import numpy as np
import scipy.optimize as scopt

from typing import Union
from .config import Config
from .profile import Profile
from pyroll.core import RollPass, Hook, DeformationUnit, root_hooks


@RollPass.Roll.extension_class
class RollExt(RollPass.Roll):
    heat_transfer_coefficient = Hook[float]()
    """Heat transfer coefficient by contact to the roll."""


@RollPass.extension_class
class RollPassExt(RollPass):
    heat_transfer_coefficient = Hook[float]()
    """Heat transfer coefficient by convection to atmosphere."""

    deformation_heat_efficiency = Hook[float]()
    """Efficiency of heat generation by deformation power."""

    environment_temperature = Hook[float]()
    """Temperature of the surrounding atmosphere."""


@RollPassExt.deformation_heat_efficiency
def deformation_heat_efficiency(self: RollPass):
    return 0.95


@RollPassExt.environment_temperature
def environment_temperature(self: RollPassExt):
    return self.roll.temperature


@RollExt.heat_transfer_coefficient
def heat_transfer_coefficient(self: RollPass):
    return 6000


@RollPassExt.heat_transfer_coefficient
def heat_transfer_coefficient(self: RollPass):
    return 150


def get_increments(unit: DeformationUnit, roll_pass: RollPassExt, ring_temperatures) -> np.ndarray:
    p: Profile = unit.in_profile

    increments = np.zeros_like(ring_temperatures)

    deformation_resistance = (
        unit.deformation_resistance
        if unit.has_value("deformation_resistance")
        else (unit.in_profile.flow_stress + 2 * unit.out_profile.flow_stress) / 3
    )

    source_density = roll_pass.deformation_heat_efficiency * deformation_resistance * unit.strain_rate

    cross_section = p.ring_sections[0].area
    increments[0] = unit.duration / (p.density * p.specific_heat_capacity * cross_section) * (
            (ring_temperatures[1] - ring_temperatures[0]) * p.ring_contours[1].length
            / p.rings[1] * p.thermal_conductivity
            + source_density * cross_section
    )

    try:
        free_surface_ratio = unit.free_surface_area / unit.surface_area
    except AttributeError:
        free_surface_ratio = 0

    cross_section = p.ring_sections[-1].area

    roll_contact_transfer = (
            roll_pass.roll.heat_transfer_coefficient * (roll_pass.roll.temperature - p.surface_temperature)
            * p.ring_contours[-1].length * (1 - free_surface_ratio)
    )

    if Config.ROLL_PASS_ATMOSPHERE_TRANSFER:
        atmosphere_transfer = (
                (
                        roll_pass.heat_transfer_coefficient
                        * (roll_pass.environment_temperature - p.surface_temperature)
                        + Config.RADIATION_COEFFICIENT * p.relative_radiation_coefficient
                        * (roll_pass.environment_temperature ** 4 - p.surface_temperature ** 4)
                )
                * p.ring_contours[-1].length * free_surface_ratio
        )
    else:
        atmosphere_transfer = 0

    increments[-1] = unit.duration / (p.density * p.specific_heat_capacity * cross_section) * (
            roll_contact_transfer + atmosphere_transfer
            - p.thermal_conductivity * (ring_temperatures[-1] - ring_temperatures[-2])
            / (p.rings[-1] - p.rings[-2])
            * p.ring_contours[-2].length
            + source_density * cross_section
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
                + source_density * cross_section
        )

    return increments


def _solve_step(unit, roll_pass, in_ring_temperatures):
    x0 = get_increments(unit, roll_pass, in_ring_temperatures)

    def f(x):
        out_ring_temperatures = in_ring_temperatures + x
        return get_increments(unit, roll_pass, out_ring_temperatures) - x

    sol = scopt.root(f, x0=x0)

    if not sol.success:
        raise RuntimeError(f"Numerical procedure did not succeed: {sol.message}.")

    return in_ring_temperatures + sol.x


@RollPass.OutProfile.ring_temperatures
def ring_temperatures_disk(self: Union[RollPass.OutProfile, Profile]):
    if not self.roll_pass.disk_elements:
        roll_pass = self.roll_pass

        return _solve_step(roll_pass, roll_pass, roll_pass.in_profile.ring_temperatures)


@RollPass.DiskElement.OutProfile.ring_temperatures
def ring_temperatures_disk(self: Union[RollPass.DiskElement.OutProfile, Profile]):
    roll_pass = self.roll_pass
    disk = self.disk_element

    return _solve_step(disk, roll_pass, disk.in_profile.ring_temperatures)


def _surface_temperature(self: Union[RollPass.Profile, Profile]):
    roll_pass: RollPassExt = self.roll_pass

    try:
        free_surface_ratio = roll_pass.free_surface_area / roll_pass.surface_area
    except AttributeError:
        free_surface_ratio = 0

    def f(ts):
        roll_contact_transfer = (
                roll_pass.roll.heat_transfer_coefficient * (roll_pass.roll.temperature - ts)
                * self.ring_contours[-1].length * (1 - free_surface_ratio)
        )

        if Config.ROLL_PASS_ATMOSPHERE_TRANSFER:
            atmosphere_transfer = (
                    (
                            roll_pass.heat_transfer_coefficient
                            * (roll_pass.environment_temperature - ts)
                            + Config.RADIATION_COEFFICIENT * self.relative_radiation_coefficient
                            * (roll_pass.environment_temperature ** 4 - ts ** 4)
                    )
                    * self.ring_contours[-1].length * free_surface_ratio
            )
        else:
            atmosphere_transfer = 0

        return (
                roll_contact_transfer + atmosphere_transfer
                - self.thermal_conductivity
                * (ts - self.ring_temperatures[-1])
                / (self.equivalent_radius - self.rings[-1])
        )

    def fprime(ts):
        return (
                - roll_pass.roll.heat_transfer_coefficient * (1 - free_surface_ratio)
                - ((
                           roll_pass.heat_transfer_coefficient
                           + 4 * Config.RADIATION_COEFFICIENT * self.relative_radiation_coefficient
                           * ts ** 3
                   ) * free_surface_ratio if Config.ROLL_PASS_ATMOSPHERE_TRANSFER else 0)
                - self.thermal_conductivity / (self.equivalent_radius - self.rings[-1])
        )

    sol = scopt.root_scalar(f=f, fprime=fprime, x0=self.ring_temperatures[-1], method="newton")

    if sol.converged:
        return sol.root


@RollPass.Profile.surface_temperature
def surface_temperature(self: Union[RollPass.Profile, Profile]):
    return _surface_temperature(self)


@RollPass.DiskElement.Profile.surface_temperature
def disk_surface_temperature(self: Union[RollPass.Profile, Profile]):
    return _surface_temperature(self)


root_hooks.add(RollPass.Profile.core_temperature)
root_hooks.add(RollPass.Profile.surface_temperature)
