import numpy as np
from pyroll.core import Profile, Unit
from pyroll.core.hooks import Hook
from pyroll.ring_model import RingProfile, RING_COUNT


@Profile.extension_class
class Profile(RingProfile):
    ring_temperatures = Hook[np.ndarray]()
    """Array temperature values from core to surface."""


@Profile.ring_temperatures
def homogeneous_profile(self: Profile):
    if self.has_set_or_cached("temperature"):
        return np.full(RING_COUNT, self.temperature)


@Unit.OutProfile.ring_temperatures
def out_ring_temperatures_from_in(self: Unit.OutProfile):
    if self.unit().in_profile.has_set_or_cached("ring_temperatures"):
        return np.copy(self.unit().in_profile.ring_temperatures)


@Profile.temperature
def mean_temperature(self: Profile):
    return np.mean(self.ring_temperatures)


@Profile.surface_temperature
def surface_temperature(self: Profile):
    return self.ring_temperatures[-1]


@Profile.core_temperature
def core_temperature(self: Profile):
    return self.ring_temperatures[0]


@Unit.OutProfile.ring_temperatures
def ring_temperatures_from_disks(self: Unit.OutProfile):
    if self.unit().subunits:
        return self.unit().subunits[-1].out_profile.ring_temperatures
