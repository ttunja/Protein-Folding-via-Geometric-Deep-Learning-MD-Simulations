"""Shared plotting style for thesis-quality figures.

Use `apply_thesis_style()` once at the top of any plotting script.
Use `smooth(arr, frac)` to smooth time series before plotting.
"""

import numpy as np
import matplotlib.pyplot as plt


THESIS_PALETTE = [
    '#1f77b4',  # blue
    '#d62728',  # red
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#17becf',  # cyan
    '#bcbd22',  # olive
]

MD_COLOR = '#222222'   # near-black for MD reference
MODEL_COLOR = '#d62728'  # red for the model in single-line plots


def apply_thesis_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#222222',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.92,
        'legend.edgecolor': '#cccccc',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.linewidth': 1.8,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 200,
        'figure.dpi': 110,
    })


def smooth(y, frac=0.03):
    """Rolling-mean smoothing. `frac` is the window as a fraction of the series length.

    Uses reflect padding to avoid edge collapse, so the output has the same length as input.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y
    w = max(3, int(round(frac * n)))
    if w >= n:
        return np.full_like(y, np.nanmean(y))
    pad = w // 2
    yp = np.concatenate([y[:pad][::-1], y, y[-pad:][::-1]])
    kernel = np.ones(w) / w
    out = np.convolve(yp, kernel, mode='same')
    # trim back to original length, keep central window
    return out[pad:pad + n]


def plot_with_smooth(ax, x, y, color, label=None, alpha_raw=0.18, alpha_smooth=0.95,
                     lw_smooth=2.0, frac=0.03, raw=True):
    """Plot one trace as light raw line + thick smoothed line on top.

    Returns the line handle of the smoothed trace (useful for legends).
    """
    if raw:
        ax.plot(x, y, color=color, alpha=alpha_raw, linewidth=0.7, zorder=1)
    ys = smooth(y, frac=frac)
    h, = ax.plot(x, ys, color=color, alpha=alpha_smooth, linewidth=lw_smooth,
                 label=label, zorder=3)
    return h
