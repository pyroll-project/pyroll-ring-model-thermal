import shapely
import numpy as np
from pyroll.report import hookimpl
from pyroll.core import Unit, PassSequence, Transport, CoolingPipe, RollPass
from pyroll.core import DiskElementUnit

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcol
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

HEATMAP_NORM = Normalize(vmin=20 + 273.15, vmax=1250 + 273.15)

@hookimpl(specname="unit_plot")
def profile_heatmap(unit: Unit):
    if isinstance(unit, RollPass | Transport):

        fig: plt.Figure = plt.figure()
        gs = gridspec.GridSpec(2, 2, height_ratios=[10, 1])

        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        cax = fig.add_subplot(gs[1, :])
        cmap = plt.get_cmap('inferno')
        norm = HEATMAP_NORM

        for ax in axs:
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        axs[0].set_title("Incoming Profile Heatmap")
        axs[1].set_title("Outgoing Profile Heatmap")

        for incoming_ring, incoming_temperature, exiting_ring, exiting_temperature in zip(reversed(unit.in_profile.ring_sections), reversed(unit.in_profile.ring_temperatures), reversed(unit.out_profile.ring_sections), reversed(unit.out_profile.ring_temperatures)):

            in_lines = incoming_ring.boundary
            out_lines = exiting_ring.boundary

            if isinstance(in_lines, shapely.MultiLineString):
                axs[0].fill(*in_lines.geoms[0].xy, color=cmap(norm(incoming_temperature)))
                axs[0].fill(*in_lines.geoms[1].xy, color='black')
            else:
                axs[0].fill(*in_lines.xy, alpha=0.5, color=cmap(norm(incoming_temperature)))

            if isinstance(out_lines, shapely.MultiLineString):
                axs[1].fill(*out_lines.geoms[0].xy, color=cmap(norm(exiting_temperature)))
                axs[1].fill(*out_lines.geoms[1].xy, color='black')
            else:
                axs[1].fill(*out_lines.xy, alpha=0.5, color=cmap(norm(exiting_temperature)))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', fraction=0.08, pad=0.12)
        cbar.set_label('Temperature [K]')

        axs[0].plot(*unit.in_profile.cross_section.boundary.xy, c="black")
        axs[1].plot(*unit.out_profile.cross_section.boundary.xy, c="black")

        fig.tight_layout()

        return fig


@hookimpl(specname="unit_plot")
def disked_unit_temperature_plot(unit: Unit):
    if isinstance(unit, DiskElementUnit):
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
