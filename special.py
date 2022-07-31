# -*- coding: utf-8 -*-
"""
Special tools for easier use of matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def set_figsize(width=None, fraction=1, subplots=(1, 1)):
    """
    Copied from: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width in ['thesis', None]:
        width_pt = 418.25555    # put \showthe\textwidth in your latex document
    elif width == 'beamer':
        width_pt = 302.0
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


# figure defaults
DPI = 400                         # default figure dpi (96, 300)
# figsize_height = 3.576811206355073     # (14, 4.5)
# figsize = (figsize_height * 16/9, figsize_height)   # default size of figures
FIGSIZE = set_figsize()

# text defaults
FONTSIZE = 10                     # fontsize
USE_TEX = False                   # use Latex fonts
DIGITS = 3                        # number of digits to round

# lines and marker defaults
LINEWIDTH = 1.0                   # linewidth (1.4)
MARKERSIZE = 1                    # marker size
MARKEREDGEWIDTH = 1               # marker edge width
COLORS = ['blue', #'darkorange', #(1, 0.15, 0),
          (1, 0.5, 0),
          'green', 'darkred', 'cyan', 'orangered', 'purple', 'lime']

# legend defaults
LEGEND_FONTSIZE = 8               # legend fontsize
LEGEND_LABEL_VSPACING = 0.2       # vertical label spacing (0.5)
LEGEND_LABEL_HSPACING = 0.3       # horizontal label spacing (0.8)
LEGEND_NUMPOINTS = 3              # number of points in legend (for marker plots)
LEGEND_EDGECOLOR = plt.rcParams["axes.facecolor"]   # edge color of the bounding box

# axes and tick defaults
AXIS_LINEWDITH = 1.0              # axes linewidth (1.2)
AXIS_LABELSIZE = 10               # label size
TICK_LABELSIZE = 8                # tick label size
MAJORSIZE = 3.5                   # size of major ticks (6.0)
MAJORWIDTH = 1.0                  # width of major ticks (1.2)
XMAJORPAD = 4.0                   # distance between xticks and their labels (9.0)
TITLEPAD = 4.0                    # distance between title and plot-Bbox (9.0)
MINORSIZE = 2.0                   # size of minor ticks (3.0)
MINORWIDTH = 0.5                  # width of minor ticks (1.0)
TICK_DIRECTION = "in"             # tick direction (in or out)
MINOR_TICK_VISIBILITY = False     # minor ticks visible


def setup_figure(dpi=DPI, figsize=FIGSIZE, colors=COLORS,
                 alw=AXIS_LINEWDITH,
                 als=AXIS_LABELSIZE,
                 majorsize=MAJORSIZE,
                 majorwidth=MAJORWIDTH,
                 xmajorpad=XMAJORPAD,
                 minorsize=MINORSIZE,
                 minorwidth=MINORWIDTH,
                 titlepad=TITLEPAD,
                 tls=TICK_LABELSIZE,
                 direction=TICK_DIRECTION,
                 visible=MINOR_TICK_VISIBILITY):
    """Sets new default parameters for figures"""
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)

    plt.rcParams["axes.linewidth"] = alw
    plt.rcParams["axes.labelsize"] = als

    plt.rcParams["axes.titlepad"] = titlepad
    plt.rcParams["xtick.major.pad"] = xmajorpad
    for axis in ["x", "y"]:
        plt.rcParams[axis + "tick.labelsize"] = tls
        plt.rcParams[axis + "tick.major.size"] = majorsize
        plt.rcParams[axis + "tick.major.width"] = majorwidth
        plt.rcParams[axis + "tick.minor.size"] = minorsize
        plt.rcParams[axis + "tick.minor.width"] = minorwidth
        plt.rcParams[axis + "tick.direction"] = direction
        plt.rcParams[axis + "tick.minor.visible"] = visible


def setup_text(fs=FONTSIZE, UseTex=USE_TEX):
    """Sets new default parameters for text"""
    plt.rcParams["font.size"] = fs

    # if 'False', we want to go back to normal fonts --> not inside 'if' !
    plt.rcParams["text.usetex"] = UseTex
    if UseTex:
        # from matplotlib import rc
        # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        # rc('font', **{'serif': ['Computer Modern']})
        plt.rcParams["font.family"] = 'serif'   # 'lmodern'
        plt.rcParams["font.serif"] = 'Computer Modern'
        plt.rcParams["text.latex.preamble"] = r'\usepackage{siunitx}'
        # plt.rcParams["text.latex.preamble"] = r'\usepackage[T1]{fontenc}'
        # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts,amssymb,amsmath,amsthm}'

        # https://stackoverflow.com/questions/17687213/how-to-obtain-the-same-
        # font-style-size-etc-in-matplotlib-output-as-in-latex


def setup_lines(
          lw=LINEWIDTH,
          ms=MARKERSIZE,
          mew=MARKEREDGEWIDTH,):
    """Sets new default parameters for lines and markers"""

    plt.rcParams["lines.linewidth"] = lw
    plt.rcParams["lines.markersize"] = ms
    plt.rcParams["lines.markeredgewidth"] = mew


def setup_legend(lfs=LEGEND_FONTSIZE,
                 lec=LEGEND_EDGECOLOR,
                 numpoints=LEGEND_NUMPOINTS,
                 lls=LEGEND_LABEL_VSPACING,
                 lhs=LEGEND_LABEL_HSPACING):
    """Sets new default parameters for legends"""

    plt.rcParams["legend.fontsize"] = lfs
    plt.rcParams["legend.numpoints"] = numpoints
    plt.rcParams["legend.edgecolor"] = lec
    plt.rcParams["legend.labelspacing"] = lls
    plt.rcParams["legend.handletextpad"] = lhs


def setup(dpi=DPI,
          figsize=FIGSIZE,
          colors=COLORS,
          alw=AXIS_LINEWDITH,
          als=AXIS_LABELSIZE,
          majorsize=MAJORSIZE,
          majorwidth=MAJORWIDTH,
          xmajorpad=XMAJORPAD,
          minorsize=MINORSIZE,
          minorwidth=MINORWIDTH,
          titlepad=TITLEPAD,
          tls=TICK_LABELSIZE,
          direction=TICK_DIRECTION,
          visible=MINOR_TICK_VISIBILITY,
          fs=FONTSIZE,
          UseTex=USE_TEX,
          lw=LINEWIDTH,
          ms=MARKERSIZE,
          mew=MARKEREDGEWIDTH,
          lfs=LEGEND_FONTSIZE,
          lec=LEGEND_EDGECOLOR,
          numpoints=LEGEND_NUMPOINTS,
          lls=LEGEND_LABEL_VSPACING,
          lhs=LEGEND_LABEL_HSPACING):
    """Sets new default parameters for plots"""
    setup_figure(dpi=dpi, figsize=figsize, colors=colors, alw=alw, als=als,
                 majorsize=majorsize, majorwidth=majorwidth,
                 xmajorpad=xmajorpad, minorsize=minorsize, minorwidth=minorwidth,
                 titlepad=titlepad, tls=tls, direction=direction, visible=visible)
    setup_text(fs=fs, UseTex=UseTex)
    setup_lines(lw=lw, ms=ms, mew=mew)
    setup_legend(lfs=lfs, lec=lec, numpoints=numpoints, lls=lls, lhs=lhs)


class Colors():
    """
    Provides a simple periodic list of colors.
    Default:
        colors == None       ->  higher contrast color cycle
        colors == 'default'  ->  default mpl color cycle

    Use 'get_color()' to get the next color in the list.
    You may pass 'inc=0' to get the same color for the next 'get_color()'
    """
    def __init__(self, colors=None, ctr=0):
        self.ctr = ctr
        if colors is None:
            self.colors = COLORS
        elif colors == "default":
            self.colors = list(plt.get_cmap("tab10").colors)
        else:
            self.colors = colors
        self.clength = len(self.colors)

    def get_color(self, inc=1):
        cval = self.colors[self.ctr % self.clength]
        self.ctr += inc
        return cval

    def prev_color(self):
        cval = self.colors[(self.ctr - 1) % self.clength]
        return cval
    

def plot_colorbar(heat, cmap='viridis', orientation="horizontal", width=0.8, height=0.3,
                  figsize=None, x0=None, y0=None, alpha=1.0, label=None):
    """
    Create a standalone colorbar based on a given 'heat'-array.
    Main functionality from ::
        https://stackoverflow.com/questions/16595138/standalone-colorbar-matplotlib
    """
    if orientation == "horizontal":
        if figsize is None:
            figsize = plt.rcParams["figure.figsize"]
            figsize = (figsize[0], 0.2 * figsize[1])
        if x0 is None:
            x0 = 0.1
        if y0 is None:
            y0 = 0.5
        if height > width:
            print(f"Warning, {orientation = } but {width = } smaller than {height = }"
                  "Values will be swapped automatically...")
            width, height = height, width

    elif orientation == "vertical":
        if figsize is None:
            figsize = plt.rcParams["figure.figsize"]
            figsize = (0.15 * figsize[0], figsize[1])
        if x0 is None:
            x0 = 0.2
        if y0 is None:
            y0 = 0.1
        if width > height:
            print(f"Warning, {orientation = } but {height = } smaller than {width = }"
                  "\nValues will be swapped automatically...")
            width, height = height, width

    plt.figure(figsize=figsize)
    plt.imshow(heat, cmap=cmap, alpha=alpha)
    plt.gca().set_visible(False)
    cax = plt.axes([x0, y0, width, height])
    plt.colorbar(orientation=orientation, cax=cax, alpha=alpha, label=label)
    plt.show()


###############################################################################
# Routines for embedding the axis-labels into the axis-ticks
###############################################################################


def set_ticks_linear(ax, vmin, vmax, numticks, decimals=7, axis='x'):
    """
    Puts 'numticks' linearly spaced ticks from 'vmin' to 'vmax' along the
        'axis' of the subplot 'ax'.
    Values are rounded to the specified 'decimals'.
    """
    ticks = np.round(np.linspace(vmin, vmax, numticks), decimals)
    getattr(ax, f"set_{axis}ticks")(ticks)
    getattr(ax, f"set_{axis}ticklabels")(ticks)


# TODO: Deprecate method in next version
def set_xticks_linear(ax, vmin, vmax, numticks, decimals=7):
    print("Deprecation Warning, use 'set_ticks_linear' with axis='x' instead.")
    set_ticks_linear(ax, vmin, vmax, numticks, decimals, axis='x')


# TODO: Deprecate method in next version
def set_yticks_linear(ax, vmin, vmax, numticks, decimals=7):
    print("Deprecation Warning, use 'set_ticks_linear' with axis='y' instead.")
    set_ticks_linear(ax, vmin, vmax, numticks, decimals, axis='y')


def ticks_in_limits(ticks, limits, DisableTicksOOB=False, axis='x'):
    """Subroutine for 'embed_labels'. Returns subset of ticks inside limits"""
    new_ticks = []
    for tick in ticks:
        tick_we = tick.get_window_extent()
        if axis == 'x':
            width = tick_we.x1 - tick_we.x0
            TickInAxis = ((tick_we.x0 + width / 2) > limits[0]
                          and (tick_we.x1 - width / 2) < limits[1])
        elif axis == 'y':
            height = tick_we.y1 - tick_we.y0
            TickInAxis = ((tick_we.y0 + height / 2) > limits[0]
                          and (tick_we.y1 - height / 2) < limits[1])
        else:
            msg = f"Wrong parameter '{axis=}', should be 'x' or 'y'."
            raise AttributeError(msg)

        if TickInAxis:
            new_ticks.append(tick)
        else:
            if DisableTicksOOB:     # set out-of-bounds ticks to be invisible
                tick.set_visible(False)
    return new_ticks

# TODO: Deprecate method in next version
def xticks_in_limits(xticks, xlimits, DisableTicksOOB=False):
    """Subroutine for 'embed_labels'. Returns subset of xticks inside limits"""
    print("Deprecation Warning, use 'ticks_in_limits' with axis='x' instead.")
    return ticks_in_limits(xticks, xlimits, DisableTicksOOB, axis='x')

# TODO: Deprecate method in next version
def yticks_in_limits(yticks, ylimits, DisableTicksOOB=False):
    """Subroutine for 'embed_labels'. Returns subset of yticks inside limits"""
    print("Deprecation Warning, use 'ticks_in_limits' with axis='y' instead.")
    return ticks_in_limits(yticks, ylimits, DisableTicksOOB, axis='y')


class AlphabeticalLabels():
    def __init__(self, abc_labels=None, ctr=0):
        if abc_labels is None:
            self.abc_labels = [r"(a)", r"(b)", r"(c)", r"(d)",
                               r"(e)", r"(f)", r"(g)", r"(h)",
                               r"(i)", r"(j)", r"(k)", r"(l)", ]
        else:
            self.abc_labels = abc_labels
        self.ctr = ctr
        self.length = len(self.abc_labels)

    def get_label(self):
        next_label = self.abc_labels[self.ctr % self.length]
        self.ctr += 1
        return next_label


def embed_labels(axes, SetCaptions=False,
                 embed_xlabels=True, embed_ylabels=True,
                 fontsize=None, labelsize=None, xva=None, yha=None,
                 DisableTicksOOB=False):
    """
    axes == single axis or list of axes on which to embed the labels

    SetCaptions == refers to enumerating the given 'axes' with (a), (b), ...

    labelAxis == list containing 'x', 'y' or 'both' for each axis from 'axes'
        refers to the axis, on which to embed the label

    fontsize == labels are replotted with the given fontsize

    xva == x-vertical alignment array with values 'top', 'center' or 'bottom'
        refers to the vertical alignment of the xlabel relative to the ticks

    yha == y-vertical alignment array with values 'right', 'center' or 'left'
        refers to the vertical alignment of the ylabel relative to the ticks

    DisableTicksOOB == Set all ticks outside of bounds to be invisible
    """
    axes = np.array([axes])
    if axes.ndim > 1:
        axes = axes.flatten()
    length = axes.shape[0]

    if fontsize is None:
        fontsize = FONTSIZE
    if labelsize is None:
        labelsize = AXIS_LABELSIZE

    if xva is None:
        xva = np.array(['center'] * length, dtype=str)
    else:
        xva = np.array([xva], dtype=str)
        if xva.shape[0] == 1:
            xva = np.full(length, xva)
    assert xva.shape[0] == length

    if yha is None:
        yha = np.array(['center'] * length, dtype=str)
    else:
        yha = np.array([yha], dtype=str)
        if yha.shape[0] == 1:
            yha = np.full(length, yha)
    assert yha.shape[0] == length

    if isinstance(SetCaptions, bool):
        if SetCaptions:
            SetCaptions = [1] * length
        else:
            SetCaptions = [0] * length
    else:
        assert len(SetCaptions) == length

    Labels = AlphabeticalLabels()

    if isinstance(embed_xlabels, bool):
        embed_xlabels = [embed_xlabels] * length
    else:
        assert len(embed_xlabels) == length

    if isinstance(embed_ylabels, bool):
        embed_ylabels = [embed_ylabels] * length
    else:
        assert len(embed_ylabels) == length

    for i, axis in enumerate(axes):
        ax0 = axis.get_window_extent().x0
        ay0 = axis.get_window_extent().y0
        ax1 = axis.get_window_extent().x1
        ay1 = axis.get_window_extent().y1
        width = ax1 - ax0
        height = ay1 - ay0

        xticks = ticks_in_limits(axis.get_xticklabels(), [ax0, ax1],
                                 DisableTicksOOB, axis='x')
        if (len(xticks) > 1) and (embed_xlabels[i] == True):
            xxpos = ((xticks[-1].get_window_extent().x0
                      + xticks[-2].get_window_extent().x1)/2 - ax0) / width
            xypos = (xticks[-1].get_window_extent().y0
                      + xticks[-1].get_window_extent().y1) / 2
            xlabel_height = (xticks[-1].get_window_extent().y1
                              - xticks[-1].get_window_extent().y0) / 2

            if xva[i] == 'bottom':
                xypos -= 2.0 * xlabel_height
            elif xva[i] == 'center':
                xypos -= xlabel_height
            elif xva[i] == 'top':
                xypos += 0.0
            else:
                msg = ("Vertical x-alignment 'xva' should have been one of "
                        + f"['top', 'center', 'bottom'], but was {xva[i]}!")
                raise ValueError(msg)

            # shift and transform to relative units
            xypos = (xypos - ay0) / height
            xlabel = axis.get_xlabel()
            axis.set_xlabel(xlabel, fontsize=labelsize, rotation=0,
                            va='center', ha='center')
            axis.xaxis.set_label_coords(xxpos, xypos)
        else:
            SetCaptions[i] = 0

        yticks = ticks_in_limits(axis.get_yticklabels(), [ay0, ay1],
                                 DisableTicksOOB, axis='y')
        if (len(yticks) > 1) and (embed_ylabels[i] == True):
            yypos = ((yticks[-1].get_window_extent().y0
                      + yticks[-2].get_window_extent().y1)/2 - ay0) / height
            yxpos = yticks[-1].get_window_extent().x1

            if yha[i] == 'left':
                yxpos = yticks[-1].get_window_extent().x0
            elif yha[i] == 'center':
                yxpos = (yticks[-1].get_window_extent().x0
                         + yticks[0].get_window_extent().x0
                         + yticks[0].get_window_extent().x1) / 3
            elif yha[i] == 'right':
                yxpos += 0
            else:
                msg = ("Horizontal y-alignment 'yha' should have been one of "
                        + f"['left', 'center', 'right'], but was {yha[i]}!")
                raise ValueError(msg)

            # shift and transform to relative units
            yxpos = (yxpos - ax0) / width
            ylabel = axis.get_ylabel()
            axis.set_ylabel(ylabel, fontsize=labelsize, rotation=0,
                            ha=yha[i], va='center')
            axis.yaxis.set_label_coords(yxpos, yypos)

            # ensure that the y-label has a minimal padding to the y-axis
            min_padding = 5
            if yxpos < 0.5:         # y-label on left axis
                if axis.yaxis.get_label().get_window_extent().x1 + min_padding > ax0:
                    axis.set_ylabel(ylabel, fontsize=labelsize, rotation=0,
                                    ha='right', va='center')
                    axis.yaxis.set_label_coords(-0.005, yypos)
                    
            elif yxpos > 0.5:         # y-label on left axis
                if axis.yaxis.get_label().get_window_extent().x0 - min_padding < ax1:
                    axis.set_ylabel(ylabel, fontsize=labelsize, rotation=0,
                                    ha='left', va='center')
                    axis.yaxis.set_label_coords(1.005, yypos)


        if SetCaptions[i]:
            xypos = (xticks[-1].get_window_extent().y0 - xlabel_height - ay0)
            xypos /= height
            axis.text(0.5, xypos, Labels.get_label(),
                      fontsize=fontsize, ha='center', va='center',
                      transform=axis.transAxes)


def polish(fig, axes, SetCaptions=False,
           embed_xlabels=True, embed_ylabels=True,
           fontsize=None, labelsize=None, xva=None, yha=None):
    """
    fig == the figure containing the relevant axes

    axes == single axis or list of axes on which to embed the labels

    SetCaptions == refers to enumerating the given 'axes' with (a), (b), ...

    labelAxis == list containing 'x', 'y' or 'both' for each axis from 'axes'
        refers to the axis, on which to embed the label

    fontsize == labels are replotted with the given fontsize

    xva == x-vertical alignment array with values 'top', 'center' or 'bottom'
        refers to the vertical alignment of the xlabel relative to the ticks

    yha == y-vertical alignment array with values 'right', 'center' or 'left'
        refers to the vertical alignment of the ylabel relative to the ticks
    """
    # FIXME: Is there no easier way of doing this?
    #        For some reason we need to 'iterate' this process...
    fig.tight_layout()
    fig.canvas.draw()
    embed_labels(axes, SetCaptions=False,
                  embed_xlabels=embed_xlabels, embed_ylabels=embed_ylabels,
                  fontsize=fontsize, labelsize=labelsize,
                  xva=xva, yha=yha, DisableTicksOOB=False)
    fig.tight_layout()
    fig.tight_layout()
    fig.tight_layout()
    fig.canvas.draw()
    embed_labels(axes, SetCaptions=SetCaptions,
                  embed_xlabels=embed_xlabels, embed_ylabels=embed_ylabels,
                  fontsize=fontsize, labelsize=labelsize,
                  xva=xva, yha=yha, DisableTicksOOB=True)
    fig.canvas.draw()
    fig.tight_layout()
    plt.show()



###############################################################################
# Provides tools for easier ticklabel manipulation
###############################################################################


def multiple_formatter(denominator=2, number=np.pi, latex=r'\pi'):
    """
    den = 2
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / den))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / (6 * den)))
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(multiple_formatter(denominator=den)))
    https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in
    -multiples-of-pi-python-matplotlib
    """
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den * x / number))
        com = gcd(num,den)
        (num, den) = (int(num / com), int(den / com))
        if den==1:
            if num == 0:
                string = r'$0$'
            elif num == 1:
                string = r'$%s$'%latex
            elif num == -1:
                string = r'$-%s$'%latex
            else:
                string = r'$%s%s$'%(num, latex)
        else:
            if num == 1:
                string = r'$\frac{%s}{%s}$'%(latex, den)
            elif num == -1:
                string = r'$-\frac{%s}{%s}$'%(latex, den)
            else:
                if num > 0:
                    string = r'$\frac{%s%s}{%s}$'%(num, latex, den)
                else:
                    string = r'$-\frac{%s%s}{%s}$'%(abs(num), latex, den)
        return string

    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex=r'\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(
            multiple_formatter(self.denominator, self.number, self.latex))


def format_ticklabels(ax, axis='x', major_den=2, minor_den=12,
                      number=np.pi, latex=r'\pi'):
    """Format the ticklabels on the axis 'axis' of the (sub)plot 'ax'. """
    if axis in ['x', 'xaxis']:
        axis = 'xaxis'
    if axis in ['y', 'yaxis']:
        axis = 'yaxis'

    subaxis = getattr(ax, axis)
    subaxis.set_major_locator(plt.MultipleLocator(number / major_den))
    subaxis.set_minor_locator(plt.MultipleLocator(number / minor_den))

    formatter = multiple_formatter(denominator=major_den,
                                   number=number, latex=latex)
    subaxis.set_major_formatter(plt.FuncFormatter(formatter))



###############################################################################
# Variants of the standard 'plot' routine
###############################################################################


def plot_lines(ax, x, y, c, cmap='viridis'):
    """
    Plot lines connecting all pairs of points in the arrays 'x' and 'y'
    with colors in the corresponding array 'c' of the same size.

    https://stackoverflow.com/questions/17240694/python-how-to-plot-one-line-in-different-colors

    Example:
        x, y = (np.random.random((100, 2)) - 0.5).cumsum(axis=0).T
        fig, ax = plt.subplots()
        plot_lines(ax, x, y, c=np.linspace(0, 1, x.shape[0]))
    """
    from matplotlib.collections import LineCollection

    # Convert format to 'segments = [[(x0,y0),(x1,y1)], [(x0,y0),(x1,y1)], ...]'
    # (-1, ...) --> size of first dimension determined automatically
    xy = np.array([x, y]).T.reshape((-1, 1, 2))
    segments = np.hstack([xy[:-1], xy[1:]])

    coll = LineCollection(segments, cmap=getattr(plt.cm, cmap))
    coll.set_array(c)           # set colors for each line segment

    ax.add_collection(coll)
    ax.autoscale_view()         # important rescaling


def plot_step(ax, x, y, PlotNaNs=False, **kwargs):
    """
    Plot 'y' over 'x' in axis 'ax' and draw horizontal lines for each pair.
    Effectively executes 'plot(new_x, new_y)' for the arrays
        new_x = [x[0], x[1], x[1], x[2], x[2], ...]
        new_y = [y[0], y[0], y[1], y[1], y[2], ...]
    """
    if PlotNaNs:   # handle NaN in y:
        y = np.copy(y)
        ylast = y[~np.isnan(y)][0]
        for i in range(y.shape[0]):
            if np.isnan(y[i]):
                y[i] = ylast
            else:
                ylast = y[i]
        
    nx = np.outer(x, np.ones(2)).flatten()[1:]
    ny = np.outer(y, np.ones(2)).flatten()[:-1]
    ax.plot(nx, ny, **kwargs)
    
    
def si_string(value, unit=r"ms", digits=DIGITS):
    r"""Returns the string generated by siunitx's command \SI{value}{unit}.
    The result is rounded to the specified number of digits.
    """
    if plt.rcParams["text.usetex"] == False:
        print(r"Warning special/si_string: \SI only possible in Latex-mode!")
        return fr"${value:.{digits}e}\,$" + unit
    return fr"\SI{{{value:.{digits}e}}}{{{unit}}}"


def main():
    print(__doc__)

    setup(UseTex=True)
    t = np.linspace(-0.05, 1.05, 200)
    # colors = Colors()
    fig, ax = plt.subplots(2, 2)
    for axis in ax.flat:
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        # axis.plot(t, np.sin(2*np.pi*t), c=colors.get_color())
        axis.plot(t, np.sin(2*np.pi*t))
    ax[1, 1].plot(t, np.cos(2*np.pi*t))


    # or use "\u00B5" for a text-mu within normal strings
    ax[0, 1].set_title(r"$\SI{}{\micro s}\si{\micro}$")
    # ax[0, 0].set_title(fr"$\SI{{{t[5]:.2f}}}{{\micro s}}\si{{\micro}}$")
    ax[0, 0].set_title(si_string(t[5], unit=r"\micro s", digits=2))
    ax[0, 1].set_xlim(-0.03, 0.9*np.pi)
    format_ticklabels(ax[0, 1], major_den=6, minor_den=24)
    ax[0, 0].axis([-0.15, 1.15, -1.78, 1.87])
    ax[1, 1].axis([0.05, 0.95, -0.95, 0.95])
    plt.subplots_adjust(left=0.07, right=0.88, top=0.98, bottom=0.08)
    polish(fig, ax, xva=['top', 'center', 'center', 'bottom'],
           yha=['left', 'center', 'center', 'right'])
    return 0

if __name__ == "__main__":
    main()
