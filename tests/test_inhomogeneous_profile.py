import pytest
import shapely
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pyroll.ring_model_thermal

from typing import Union
from pyroll.core import Profile
from pyroll.ring_model import RingProfile


@pytest.mark.parametrize(
    "p", [
        Profile.round(radius=10, core_temperature=973.15, surface_temperature=1273.15),
        Profile.square(side=10, corner_radius=1, core_temperature=973.15, surface_temperature=1273.15),
        Profile.box(height=10, width=5, corner_radius=1, core_temperature=973.15, surface_temperature=1273.15),
        Profile.diamond(height=5, width=10, corner_radius=1, core_temperature=973.15, surface_temperature=1273.15),

        Profile.round(radius=10, core_temperature=1273.15, surface_temperature=1073.15),
        Profile.square(side=10, corner_radius=1, core_temperature=1273.15, surface_temperature=973.15),
        Profile.box(height=10, width=5, corner_radius=1, core_temperature=1273.15, surface_temperature=973.15),
        Profile.diamond(height=5, width=10, corner_radius=1, core_temperature=1273.15, surface_temperature=973.15)
    ]
)
def test_inhomogeneous_in_profile(p: Union[RingProfile, Profile]):
    try:
        pyroll.ring_model.Config.RING_COUNT = 15
        fig: plt.Figure = plt.figure()
        axs: list[plt.Axes] = fig.subplots(nrows=2)
        axs[0].set_aspect("equal")
        cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=min(p.ring_temperatures), vmax=max(p.ring_temperatures))

        for ring, temperature in zip(reversed(p.ring_sections), reversed(p.ring_temperatures)):
            lines = ring.boundary
            if isinstance(lines, shapely.MultiLineString):
                axs[0].fill(*lines.geoms[0].xy, color=cmap(norm(temperature)))
                axs[0].fill(*lines.geoms[1].xy, color='black')
            else:
                axs[0].fill(*lines.xy, alpha=0.5, color=cmap(norm(temperature)))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axs[0], label='Temperature')

        axs[0].plot(*p.cross_section.boundary.xy, c="black")

        axs[1].plot(p.rings, p.ring_temperatures)

        fig.tight_layout()
        plt.show()

    finally:
        del pyroll.ring_model.Config.RING_COUNT
