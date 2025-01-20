#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module provides plotting utilities for:
1. Explained Variance (bar + step plot).
2. EOF/Correlation Maps via Cartopy.
3. Principal Component (PC) time series for multiple EOF modes.
4. Spatial maps with threshold-based hatching.
5. Taylor Diagram for comparing model output vs reference data.

Author: Kachinga Silwimba
Date:   2025-01-07
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point


# =============================================================================
#                     1. EXPLAINED VARIANCE PLOT
# =============================================================================

def plot_explained_variance(num_subplots, var_fracs_list, title_list, n_modes=10):
    """
    Plot individual and cumulative explained variance for multiple sets of EOF results.

    Parameters
    ----------
    num_subplots : int
        Number of subplots to create (e.g., number of experiments or models).
    var_fracs_list : list of 1D arrays
        Each entry is a 1D array (or xarray.DataArray) of variance fractions.
    title_list : list of str
        Titles for each subplot, matching the order of var_fracs_list.
    n_modes : int, optional
        Number of EOF modes to display (default=10).
    """
    rows, cols = 2, 2  # 2x2 grid
    plt.figure(figsize=(6 * cols, 5 * rows))

    for i in range(num_subplots):
        ax = plt.subplot(rows, cols, i + 1)

        var_fracs = var_fracs_list[i]
        cum_sum_exp = np.cumsum(var_fracs)

        # Bar chart for individual explained variance
        ax.bar(
            range(n_modes),
            var_fracs[:n_modes] * 100,
            alpha=0.7,
            align='center',
            color="red",
            label='Individual explained variance'
        )
        ax.set_ylabel('Individual Var. Expl. [%]', fontweight='regular', fontsize=13)
        ax.set_ylim(0, max(var_fracs[:n_modes] * 100) + 1)
        ax.set_xlabel('EOF index', fontweight='regular', fontsize=11)
        ax.set_title(title_list[i], pad=15, fontweight='regular', fontsize=11)
        ax.legend(loc="upper left", fontsize=11)
        ax.minorticks_on()

        # Step plot for cumulative explained variance on a secondary y-axis
        ax2 = ax.twinx()
        ax2.step(
            range(n_modes),
            cum_sum_exp[:n_modes] * 100,
            where='mid',
            color="green",
            label='Cumulative explained variance'
        )
        ax2.set_ylabel('Cumulative Expl. var. [%]', fontweight='regular', fontsize=10)
        ax2.legend(loc="center", fontsize=11)
        ax2.minorticks_on()

    plt.tight_layout()


# =============================================================================
#                     2. EOF/CORRELATION MAPS
# =============================================================================

def contour_plot(eofs, var_fracs, ax, title, is_first_subplot=False, is_last_row=False):
    """
    Create a spatial pcolormesh plot for EOF or correlation data on a GeoAxes.

    Parameters
    ----------
    eofs : xarray.DataArray or numpy.array
        2D array of shape (lat, lon) containing EOF or correlation values.
    var_fracs : xarray.DataArray or float
        Variance fraction (or scalar) for annotation on the subplot.
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Axes on which to draw the map.
    title : str
        Subplot title.
    is_first_subplot : bool, optional
        If True, shows y-axis labels/ticks on the left side.
    is_last_row : bool, optional
        If True, shows x-axis labels/ticks on the bottom.

    Returns
    -------
    mesh : matplotlib.collections.QuadMesh
        The pcolormesh instance for further customization.
    """
    # Add a cyclic point to avoid the "seam" issue at 0/360 degrees
    data, lons = add_cyclic_point(eofs, coord=eofs['lon'])
    lats = eofs['lat']
    lons2d, lats2d = np.meshgrid(lons, lats)

    # Pcolormesh for the field
    mesh = ax.pcolormesh(
        lons2d, lats2d, data,
        transform=ccrs.PlateCarree(),
        cmap='RdYlBu_r',  # diverging colormap
        shading='auto'
    )

    # Add state/province boundaries
    states = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none'
    )
    ax.add_feature(states, edgecolor='gray')

    # Set subplot title
    ax.set_title(title, fontweight='regular', fontsize=11, loc='left')

    # Annotate with the variance fraction
    frac_value = var_fracs.values if hasattr(var_fracs, 'values') else var_fracs
    ax.text(
        0.95, 0.05,
        f'VE: {frac_value * 100:.2f}%',
        transform=ax.transAxes,
        fontsize=9,
        fontweight='regular',
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
    )

    # Set latitude/longitude ticks
    if is_last_row:
        ax.set_xticks(np.arange(-120, -60, 10), crs=ccrs.PlateCarree())
        lon_formatter = cticker.LongitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)

    if is_first_subplot:
        ax.set_yticks(np.arange(24, 50, 5), crs=ccrs.PlateCarree())
        lat_formatter = cticker.LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)

    ax.minorticks_on()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle='-')
    ax.add_feature(cfeature.COASTLINE)

    return mesh


# =============================================================================
#                     3. PLOT PC TIME SERIES
# =============================================================================

colors = ['red', 'green', 'black', 'blue', 'cyan', 'yellow', 'coral']
def plot_eofs_over_time(ax, years, eof_amplitudes, eof_labels, title, xlabel, ylabel, colors=colors, x_label=False):
    """
    Plot multiple EOF principal component (PC) time series on the same axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot.
    years : array-like
        X-values (typically years or time indices).
    eof_amplitudes : list of arrays
        Each array corresponds to one EOF's PC time series.
    eof_labels : list of str
        Labels for each EOF's PC series, for the legend.
    title : str
        Plot title.
    xlabel : str
        Label for the x-axis (shown only if x_label=True).
    ylabel : str
        Label for the y-axis.
    colors : list of str, optional
        Colors used for each PC series (default = pre-defined list).
    x_label : bool, optional
        If True, the x-axis label is shown; otherwise, it's hidden (useful for subplots).

    Returns
    -------
    None
    """
    lines = []  # To store line objects for the legend

    for i, amp in enumerate(eof_amplitudes):
        label = eof_labels[i] if eof_labels else None
        line, = ax.plot(years, amp, label=label, linestyle="-", linewidth=1.3,
                        color=colors[i % len(colors)])
        lines.append(line)

    # X/Y labels and title
    ax.set_ylabel(ylabel, fontweight='regular', fontsize=10)
    ax.set_title(title, fontweight='regular', loc='left', size=12)

    # Add legend if labels are given
    if eof_labels:
        legend = ax.legend(handles=lines, loc="upper left", fontsize=7, borderpad=1, frameon=True)
        legend.get_frame().set_linewidth(1)
        legend.get_frame().set_edgecolor("black")
        legend.get_title().set_fontweight('regular')
        legend.get_title().set_fontsize(10)

    # Format the y-axis for scientific notation
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.axhline(0, color='k')
    ax.minorticks_on()

    # Optionally set the x-axis label
    if x_label:
        ax.set_xlabel(xlabel, fontweight='regular', fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


# =============================================================================
#                     4. EUCLIDEAN DISTANCE SPATIAL MAP WITH THRESHOLD
# =============================================================================

def plot_map_with_threshold(ax, distance, threshold, mask, lon, lat, title,
                            is_first_subplot=False, is_last_row=False):
    """
    Plot a 2D field (e.g., Euclidean distance) and overlay hatches
    for regions under a threshold.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis on which to plot.
    distance : 2D array
        The field to be plotted (lat x lon).
    threshold : float
        Threshold value for applying hatching.
    lon : 1D array
        Longitudes for the data.
    lat : 1D array
        Latitudes for the data.
    title : str
        Subplot title.
    is_first_subplot : bool, optional
        If True, shows y-axis labels/ticks.
    is_last_row : bool, optional
        If True, shows x-axis labels/ticks.

    Returns
    -------
    mesh : matplotlib.collections.QuadMesh
        Pcolormesh instance for further customization.
    """
    # Example usage of 'mask' (assuming it's defined globally)
    data_masked = distance * mask

    # Add a cyclic point for the 0/360 seam
    data_cyclic, lons_cyclic = add_cyclic_point(data_masked, coord=lon)
    lons2d, lats2d = np.meshgrid(lons_cyclic, lat)

    # Create filled pcolormesh
    mesh = ax.pcolormesh(
        lons2d, lats2d, data_cyclic,
        transform=ccrs.PlateCarree(),
        cmap='YlOrRd',
        shading='auto'
    )

    # Create a boolean mask for hatching (where data < threshold)
    hatch_mask = np.ma.masked_less(data_cyclic, threshold)

    # Overlay hatches
    ax.contourf(
        lons_cyclic,
        lat,
        hatch_mask.mask,
        levels=[0.5, 1],
        colors='none',
        hatches=['//'],
        transform=ccrs.PlateCarree()
    )

    # Add map features
    states = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none'
    )
    ax.add_feature(states, edgecolor='gray')
    ax.set_title(title, loc='left', fontweight='regular', fontsize=10)
    ax.minorticks_on()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle='-')
    ax.add_feature(cfeature.COASTLINE)

    # Latitude/Longitude ticks
    if is_last_row:
        ax.set_xticks(np.arange(-120, -60, 10), crs=ccrs.PlateCarree())
        lon_formatter = cticker.LongitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)

    if is_first_subplot:
        ax.set_yticks(np.arange(24, 50, 5), crs=ccrs.PlateCarree())
        lat_formatter = cticker.LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)

    ax.minorticks_on()
    return mesh


# =============================================================================
#                     5. TAYLOR DIAGRAM
# =============================================================================

import numpy as NP
import matplotlib.pyplot as PLT

def create_taylor_diagram(stdref, samples, title, fig, subplot_num, add_legend=False):
    """
    Create a Taylor Diagram comparing standard deviations & correlations vs a reference.

    Parameters
    ----------
    stdref : float
        Reference standard deviation (ERA5 or other reference).
    samples : list of tuples
        Each tuple is (sample_stddev, corrcoef, name_str).
    title : str
        Title for this Taylor Diagram subplot.
    fig : matplotlib.figure.Figure
        The figure to which we add the subplot.
    subplot_num : int or 3-digit code
        E.g., 131, 132, 133 for 3-subplot layout, or an integer to specify the subplot.
    add_legend : bool, optional
        If True, adds a legend for the sample points in the diagram.

    Returns
    -------
    dia : TaylorDiagram
        The created TaylorDiagram object for further customization.
    """
    # Initialize the diagram in the given subplot
    dia = TaylorDiagram(stdref, fig=fig, rect=subplot_num, label='ERA5', extend=False)
    dia.samplePoints[0].set_color('r')  # Red star (reference point)

    # Define markers and colors for up to 8 samples
    markers = ['o', '^', 's', 'D', 'P', 'h', 'X', '*']
    colors = ['b', 'g', 'c', 'm', 'black', 'y', 'coral']

    # Add each sample
    for i, (stddev, corrcoef, name) in enumerate(samples):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        dia.add_sample(stddev, corrcoef, marker=marker, ms=10, ls='',
                       mfc=color, mec=color, label=name)

    # Add RMS contours
    contours = dia.add_contours(levels=8, colors='0.5', linewidths=0.5)
    plt.clabel(contours, inline=1, fontsize=12, fmt='%.1f')

    # Grid
    dia._ax.grid(lw=0.9, alpha=0.9)

    # Optional legend
    if add_legend:
        fig.legend(
            [p for p in dia.samplePoints],
            [p.get_label() for p in dia.samplePoints],
            numpoints=1, prop=dict(size='small'), loc=(0.27, 0.65)
        )

    dia._ax.set_title(title, size='large', fontweight="bold")
    fig.tight_layout()
    return dia

