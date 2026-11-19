"""Shared matplotlib style: Times New Roman + the 'Macaron' palette.

Times New Roman is used when it is installed (Windows/macOS); otherwise a
metric-compatible Times clone (TeX Gyre Termes / Liberation Serif). Font files
dropped into <project>/fonts/ are registered automatically.
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# --- Macaron palette -------------------------------------------------------
CORAL = "#F28E8E"     # primary accent  (our method)
SKY = "#8EC5E8"       # cool accent
MINT = "#95D5B2"      # fresh accent
PEACH = "#F7C59F"     # warm accent
LAVENDER = "#C3AED6"  # soft purple
BUTTER = "#F3D675"    # highlight
AQUA = "#7FC8C2"      # secondary accent
ROSE = "#E7A1C4"      # soft contrast
SOFT_BLACK = "#2D2D32"  # text / axes
WARM_WHITE = "#FAFAF7"  # background
GRAY = "#92979F"        # grid / inactive
BROWN = "#8B6F5A"       # border / annotation

# categorical order: index 0 is always "our method"
C = [CORAL, SKY, MINT, LAVENDER, PEACH, AQUA, ROSE, BUTTER]
INK, INK2, MUTED, GRID, AXIS, SURFACE = SOFT_BLACK, SOFT_BLACK, GRAY, GRAY, SOFT_BLACK, WARM_WHITE
DIV_NEG, DIV_MID, DIV_POS = SKY, WARM_WHITE, CORAL

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _register_fonts():
    for f in glob.glob(os.path.join(_ROOT, "fonts", "*.tt[fc]")) + glob.glob(os.path.join(_ROOT, "fonts", "*.otf")):
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            pass


def serif_family():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Times New Roman", "TeX Gyre Termes", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]:
        if cand in names:
            return cand
    return "serif"


TEXTWIDTH = 6.5  # inches: text width of the report (US letter, 1in margins); figures are drawn at final size


def setup():
    """Global style. Figures are created at their printed size (width <= TEXTWIDTH) and included at 100%,
    so every figure in the report has the same font sizes."""
    _register_fonts()
    fam = serif_family()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [fam, "Times New Roman", "TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # Times-like mathematics
        "font.size": 8,
        "text.color": SOFT_BLACK,
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.04,
        "figure.constrained_layout.w_pad": 0.04,
        "figure.constrained_layout.hspace": 0.06,
        "figure.constrained_layout.wspace": 0.06,
        "savefig.dpi": 300,
        "savefig.bbox": None,  # keep the exact figure size so that fonts print at their nominal size
        "savefig.facecolor": "white",
        "axes.facecolor": WARM_WHITE,
        "axes.edgecolor": SOFT_BLACK,
        "axes.linewidth": 0.6,
        "axes.labelcolor": SOFT_BLACK,
        "axes.labelsize": 8,
        "axes.labelpad": 2.5,
        "axes.titlecolor": SOFT_BLACK,
        "axes.titlesize": 8.5,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.titlepad": 4,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRAY,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.45,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": SOFT_BLACK,
        "ytick.color": SOFT_BLACK,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.3,
        "legend.columnspacing": 1.2,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.2,
        "lines.markeredgecolor": SOFT_BLACK,
        "lines.markeredgewidth": 0.45,
        "errorbar.capsize": 1.8,
        "patch.edgecolor": SOFT_BLACK,
        "patch.linewidth": 0.5,
        "axes.prop_cycle": matplotlib.cycler(color=C),
        "pdf.fonttype": 42,
    })
    return fam


def heatmap(ax, V, cmap, norm):
    """Vector heat map (one filled rectangle per cell; no embedded raster image), rows from top to bottom."""
    import numpy as np
    R, C = V.shape
    ax.pcolormesh(np.arange(C + 1) - 0.5, np.arange(R + 1) - 0.5, V, cmap=cmap, norm=norm, shading="flat",
                  edgecolors="face", linewidth=0.0, rasterized=False)
    ax.set_xlim(-0.5, C - 0.5)
    ax.set_ylim(R - 0.5, -0.5)


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("macaron_div", [DIV_NEG, DIV_MID, DIV_POS])


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("macaron_seq", [WARM_WHITE, CORAL])
