import numpy as np
import scipy.special as sp

from scipy import interpolate
from pyroll.core import Profile, Unit
from pyroll.core.hooks import Hook
from pyroll.ring_model import RingProfile


@Profile.extension_class
class Profile(RingProfile):
    ring_temperatures = Hook[np.ndarray]()
    """Array temperature values from core to surface."""

    relative_radiation_coefficient = Hook[float]()
    """Heat transfer coefficient by convection to atmosphere."""


@Profile.relative_radiation_coefficient
def relative_radiation_coefficient(self: Profile):
    return 0.8


@Profile.ring_temperatures
def homogeneous_profile(self: Profile):
    if self.has_set_or_cached("temperature"):
        return np.full_like(self.rings, self.temperature)


@Profile.ring_temperatures
def approximated_quadratic_profile(self: Profile):
    if self.has_set_or_cached("core_temperature") and self.has_set_or_cached("surface_temperature"):
        normalized_radii = np.linspace(0, 1, len(self.rings))
        return self.core_temperature + (self.surface_temperature - self.core_temperature) * normalized_radii ** 2


@Unit.OutProfile.ring_temperatures
def out_ring_temperatures_from_in(self: Unit.OutProfile):
    if self.unit.in_profile.has_set_or_cached("ring_temperatures"):
        return np.copy(self.unit.in_profile.ring_temperatures)


@Profile.temperature
def mean_temperature(self: Profile):
    return np.sum(self.ring_temperatures * [s.area for s in self.ring_sections]) / np.sum(
        [s.area for s in self.ring_sections])


@Profile.surface_temperature
def surface_temperature(self: Profile):
    return self.ring_temperatures[-1]


@Profile.core_temperature
def core_temperature(self: Profile):
    return self.ring_temperatures[0]


@Unit.OutProfile.ring_temperatures
def ring_temperatures_from_disks(self: Unit.OutProfile):
    if self.unit.subunits:
        return self.unit.subunits[-1].out_profile.ring_temperatures
