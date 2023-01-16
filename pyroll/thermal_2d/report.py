from pyroll.report import hookimpl
from pyroll.core import Unit
import matplotlib.pyplot as plt


@hookimpl(specname="unit_plot")
def temperature_profile_plot(unit: Unit):
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.subplots()

    ax.plot(unit.in_profile.temperature_profile[0], unit.in_profile.temperature_profile[1])
    ax.plot(unit.out_profile.temperature_profile[0], unit.out_profile.temperature_profile[1])

    return fig
