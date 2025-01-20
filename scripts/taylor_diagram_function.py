#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module provides:
1. A custom TaylorDiagram class for comparing model vs. reference
   standard deviations and correlation coefficients on a single-quadrant plot.
2. A correlation function (corrcoef) for computing Pearson correlation 
   between flattened xarray.DataArrays.
3. A 'create_taylor_diagram' function for constructing individual
   Taylor Diagrams in a multi-subplot figure.

Author: Kachinga Silwimba
Date:   2025-01-07

Usage:
- Instantiate the 'TaylorDiagram' class with a reference standard deviation.
- Use 'create_taylor_diagram' to add model samples (stddev, corrcoef, label).
- The script assumes that the reference data is ERA5 and the models are
  CLM5 or other experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kendalltau

# =============================================================================
# 1. TAYLOR DIAGRAM CLASS
# =============================================================================

class TaylorDiagram(object):
    """
    Taylor diagram for visualizing how well model fields match
    a reference field in terms of their standard deviation and
    pattern correlation.

    Reference:
    - Based on https://gist.github.com/ycopin/3342888 (by Nicolas C. 
      for MPL versions).
    - Single-quadrant polar plot: radius = stddev, angle = arccos(corr).

    Parameters
    ----------
    refstd : float
        Reference standard deviation (e.g., from ERA5).
    fig : matplotlib.figure.Figure or None
        Existing figure to which the diagram is added. If None, a new
        figure is created.
    rect : int or str
        Subplot definition (e.g., 111 for a single subplot).
    label : str
        Label for the reference point (default '_').
    srange : tuple of floats
        (min_factor, max_factor) to set the radial (stddev) axis range as
        [min_factor * refstd, max_factor * refstd].
    extend : bool
        If True, the plot extends to negative correlations (theta up to pi).
        Otherwise, positive correlations only (theta up to pi/2).

    Attributes
    ----------
    ax : mpl_toolkits.axisartist.floating_axes.FloatingAxes
        The polar coordinates axis for plotting.
    samplePoints : list
        A collection of line objects corresponding to added samples.

    Examples
    --------
    >>> # Suppose stdref is the reference standard deviation
    >>> td = TaylorDiagram(stdref=1.0, fig=plt.figure(), rect=111, label='REF')
    >>> # Add a sample point
    >>> td.add_sample(stddev=0.9, corrcoef=0.95, marker='o', label='Model1')
    >>> td.add_contours(levels=5, colors='0.5')  # RMS contours
    """

    def __init__(self, refstd, fig=None, rect=111, label='_', srange=(0, 1.5), extend=False):
        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd  # Reference standard deviation

        # Define polar transform
        tr = PolarAxes.PolarTransform()

        # Correlation tick labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Full diagram for negative correlations
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi / 2

        # Convert correlation ticks to angles
        tlocs = np.arccos(rlocs)

        # Build grid locators/formatters
        gl1 = GF.FixedLocator(tlocs)                # angles
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Radius extent for stddev axis
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        # Create grid helper
        ghelper = FA.GridHelperCurveLinear(
            tr, 
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1,
            tick_formatter1=tf1
        )

        if fig is None:
            fig = plt.figure()

        # Create a floating subplot
        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Angle axis (top)
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Pattern Correlation")

        # X axis (left)
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard Deviation")

        # Y axis (right)
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left"
        )

        # Bottom axis (unused if smin != 0)
        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)

        # Assign references to the axis objects
        self._ax = ax                  # AxisArtist object
        self.ax = ax.get_aux_axes(tr)  # Polar transform axis

        # Plot reference point
        l_ref, = self.ax.plot([0], self.refstd, 'k*', ls='', ms=10, label=label)

        # Draw reference stddev circle
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')  # underscore avoids legend entry

        # Keep track of sample points
        self.samplePoints = [l_ref]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add a model sample (stddev, corrcoef) on the diagram.

        Parameters
        ----------
        stddev : float
            Model standard deviation.
        corrcoef : float
            Correlation coefficient with the reference.
        *args, **kwargs : 
            Passed to matplotlib plot function (marker, color, etc.).

        Returns
        -------
        line : matplotlib.lines.Line2D
            The line object representing this sample point.
        """
        # Convert correlation to angle, stddev to radius
        angle = np.arccos(corrcoef)
        line, = self.ax.plot(angle, stddev, *args, **kwargs)
        self.samplePoints.append(line)
        return line

    def add_grid(self, *args, **kwargs):
        """Add grid lines to the main axis."""
        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add RMS difference contours.

        Parameters
        ----------
        levels : int or list
            Number of contour levels or explicit level boundaries.
        **kwargs : 
            Passed to matplotlib contour function.

        Returns
        -------
        contours : QuadContourSet
            The matplotlib contour set.
        """
        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax),
            np.linspace(0, self.tmax)
        )
        # Centered RMS difference formula
        rms = np.sqrt(
            self.refstd**2 + rs**2 - 2*self.refstd * rs * np.cos(ts)
        )
        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return contours


# =============================================================================
# 2. EOF CORRELATION FUNCTION
# =============================================================================

def corrcoef(eofmode_era5, eofmode_spmip):
    """
    Flatten and correlate two 2D EOF spatial modes.

    Parameters
    ----------
    eofmode_era5 : xarray.DataArray
        EOF pattern from ERA5 (2D: lat x lon).
    eofmode_spmip : xarray.DataArray
        EOF pattern from a model experiment (2D: lat x lon).

    Returns
    -------
    float
        Pearson correlation coefficient between the flattened, non-NaN
        values of 'eofmode_era5' and 'eofmode_spmip'.

    Notes
    -----
    - NaN values are removed before correlation.
    """
    era5_arr = eofmode_era5.values.flatten()
    spmip_arr = eofmode_spmip.values.flatten()

    # Mask out NaNs
    mask = ~np.isnan(era5_arr) & ~np.isnan(spmip_arr)
    era5_valid = era5_arr[mask]
    spmip_valid = spmip_arr[mask]

    # Calculate the Pearson correlation
    # (np.corrcoef returns a 2x2 matrix)
    return np.corrcoef(era5_valid, spmip_valid)[0, 1]


# =============================================================================
# 3. TAYLOR DIAGRAM PLOTTING FUNCTION
# =============================================================================

def create_taylor_diagram(stdref, samples, title, fig, subplot_num, add_legend=False):
    """
    Create a single Taylor Diagram subplot for a specific EOF or variable.

    Parameters
    ----------
    stdref : float
        Reference standard deviation (e.g., from ERA5).
    samples : list of tuples
        Each tuple: (model_stddev, model_corr, label_str). Example:
        [(0.8, 0.9, 'EXP1'), (1.2, 0.88, 'EXP2'), ...]
    title : str
        Subplot title, e.g. '[a] EOF-1'.
    fig : matplotlib.figure.Figure
        The figure onto which this subplot is drawn.
    subplot_num : int or str
        Subplot code, e.g. 131 for a 1×3 grid, left subplot.
    add_legend : bool, optional
        Whether to add a legend showing model labels.

    Returns
    -------
    dia : TaylorDiagram
        The created TaylorDiagram object for further customization.

    Examples
    --------
    >>> fig = plt.figure(figsize=(10,4))
    >>> samples = [(0.9, 0.94, 'EXP1'), (1.1, 0.88, 'EXP2')]
    >>> td = create_taylor_diagram(1.0, samples, '[a] EOF1', fig, 131, add_legend=True)
    >>> plt.show()
    """
    # Instantiate the Taylor Diagram
    dia = TaylorDiagram(stdref, fig=fig, rect=subplot_num, label='ERA5', extend=False)

    # Color the reference point red
    dia.samplePoints[0].set_color('r')

    # Markers and colors for up to 8 samples
    markers = ['o', '^', 's', 'D', 'P', 'h', 'X', '*']
    colors = ['b', 'g', 'c', 'm', 'black', 'y', 'coral']

    # Add each sample to the diagram
    for i, (stddev, corr, name) in enumerate(samples):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        dia.add_sample(
            stddev,
            corr,
            marker=marker,
            ms=10,
            ls='',
            mfc=color,
            mec=color,
            label=name
        )

    # Add RMS difference contours
    contours = dia.add_contours(levels=8, colors='0.5', linewidths=0.5)
    plt.clabel(contours, inline=1, fontsize=12, fmt='%.1f')

    # Add grid
    dia._ax.grid(lw=0.9, alpha=0.9)

    # Optional legend
    if add_legend:
        fig.legend(
            [p for p in dia.samplePoints],
            [p.get_label() for p in dia.samplePoints],
            numpoints=1, prop=dict(size='small'),
            loc=(0.27, 0.65)  # Adjust location if needed
        )

    # Title
    dia._ax.set_title(title, size='large', fontweight='bold')

    # Adjust layout
    fig.tight_layout()

    return dia
