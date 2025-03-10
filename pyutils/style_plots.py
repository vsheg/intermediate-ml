import matplotlib as mpl

custom_rc = {
    # Axes
    "axes.titlesize": "large",
    "axes.labelsize": "large",
    "axes.formatter.use_mathtext": True,
    # Lines
    "lines.linewidth": 1.0,
    "lines.markersize": 5,
    "hatch.linewidth": 0.25,
    "patch.antialiased": True,
    # Ticks
    "xtick.top": False,
    "xtick.bottom": False,
    "xtick.minor.visible": True,
    "xtick.labelsize": "large",
    "ytick.left": False,
    "ytick.right": False,
    "ytick.minor.visible": True,
    "ytick.labelsize": "large",
    # Legend
    "legend.fontsize": 9,
    # Figure size
    "figure.figsize": (5.0, 3.0),
    "figure.subplot.left": 0.125,
    "figure.subplot.bottom": 0.175,
    "figure.subplot.top": 0.95,
    "figure.subplot.right": 0.95,
    "figure.autolayout": False,
    # Fonts
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"],
    # Saving figures
    "path.simplify": True,
    #
    #
    #
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.frameon": False,
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "axes.grid.which": "both",
}


def style_plots():
    mpl.rcParams.update(custom_rc)
