import numpy as np
from pyroll.report import hookimpl
from pyroll.core import Unit, PassSequence
from pyroll.core.disk_element import DiskedUnit
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcol


@hookimpl(specname="unit_plot")
def disked_unit_temperature_plot(unit: Unit):
    if isinstance(unit, DiskedUnit):
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        count = len(unit.subunits) + 2
        cmap = mpl.colormaps['coolwarm_r']
        colors = cmap(np.linspace(0, 1, count))

        def prepare_data(p):
            return (
                np.append(p.rings, p.equivalent_radius) / p.equivalent_radius,
                np.append(p.ring_temperatures, p.surface_temperature)
            )

        ip = ax.plot(*prepare_data(unit.in_profile), c=colors[0], label="incoming profile")[0]

        for i in range(1, count - 1):
            u = unit.subunits[i - 1]
            ax.plot(*prepare_data(u.out_profile), c=colors[i])

        op = ax.plot(*prepare_data(unit.out_profile), c=colors[-1], label="outgoing profile")[0]

        ax.set_title("Radial Temperature Profile Evolution")
        ax.set_xlabel("Relative Distance from Core")
        ax.set_ylabel("Temperature")

        lc = mcol.LineCollection(
            5 * [[(0, 0)]],
            colors=cmap(np.linspace(0, 1, 7)[1:-1]),
            label="intermediate stages",
        )

        ax.legend(handles=[ip, lc, op], handler_map={type(lc): HandlerDashedLines()})

        return fig


@hookimpl(specname="unit_plot")
def pass_sequence_temperature_plot(unit: Unit):
    if isinstance(unit, PassSequence):
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()

        def yield_data(u: Unit, name):
            yield getattr(u.in_profile, name, None)
            if u.subunits:
                for su in u.subunits:
                    yield from yield_data(su, name)
            yield getattr(u.out_profile, name, None)

        t = np.array(list(yield_data(unit, "t")))
        core = np.array(list(yield_data(unit, "core_temperature")))
        surface = np.array(list(yield_data(unit, "surface_temperature")))
        mean = np.array(list(yield_data(unit, "temperature")))

        ax.plot(t, core, label="core")
        ax.plot(t, surface, label="surface")
        ax.plot(t, mean, label="mean")

        ax.legend()

        ax.set_title("Temperature Evolution")
        ax.set_xlabel("Cumulative Process Time $t$")
        ax.set_ylabel("Temperature $T$")

        return fig


# from https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html#sphx-glr-gallery-text-labels-and-annotations-legend-demo-py
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.lines import Line2D


class HandlerDashedLines(HandlerLineCollection):
    """
    Custom Handler for LineCollection instances.
    """

    def create_artists(
            self, legend, orig_handle,
            xdescent, ydescent, width, height, fontsize, trans
    ):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(
            legend, xdescent, ydescent,
            width, height, fontsize
        )
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = np.full_like(xdata, height / (numlines + 1))
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[1] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines
