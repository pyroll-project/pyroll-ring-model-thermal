import numpy as np
from pyroll.core import Profile, Unit
from pyroll.core.hooks import Hook
from .constants import RADIAL_DISCRETIZATION_COUNT


@Profile.extension_class
class Profile(Profile):
    equivalent_radius = Hook[float]()
    """Radius of an equivalent round profile."""

    temperature_profile = Hook[np.ndarray]()
    """2D Array of radius coordinates and related temperature values from core to surface."""


@Profile.equivalent_radius
def equivalent_radius(self: Profile):
    return np.sqrt(self.cross_section.area / np.pi)


@Profile.temperature_profile
def homogeneous_profile(self: Profile):
    if self.has_set_or_cached("temperature"):
        return np.array([
            np.linspace(0, self.equivalent_radius, RADIAL_DISCRETIZATION_COUNT),
            np.full(RADIAL_DISCRETIZATION_COUNT, self.temperature)
        ])


@Unit.OutProfile.temperature_profile
def out_temperature_profile_from_in(self: Unit.OutProfile):
    if self.unit().in_profile.has_set_or_cached("temperature_profile"):
        return np.copy(self.unit().in_profile.temperature_profile)


@Profile.temperature
def mean_temperature(self: Profile):
    # if self.has_value("temperature_profile"):
    return np.mean(self.temperature_profile[1])


@Profile.surface_temperature
def surface_temperature(self: Profile):
    # if self.has_value("temperature_profile"):
    return self.temperature_profile[1, -1]


@Profile.core_temperature
def core_temperature(self: Profile):
    # if self.has_value("temperature_profile"):
    return self.temperature_profile[1, 0]
