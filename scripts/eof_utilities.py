#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Kachinga Silwimba
Date:   2025-01-07
"""

# =============================================================================
#                           EOFs FUNCTION SCRIPT
# =============================================================================

"""
This script provides a suite of functions for working with Empirical Orthogonal Functions (EOFs).
It includes utilities for calculating anomalies, performing EOF analysis, and projecting data onto EOF patterns.

Dependencies:
-------------
- xarray
- numpy
- eofs (EOF analysis library)

Functions:
----------
1. calculate_anomaly: Compute anomalies relative to the mean annual cycle.
2. eofcalc_pcnorm: Perform EOF analysis while normalizing the principal components (PCs) to unit variance.
3. proj_onto_eof: Project a data array onto a pre-computed EOF pattern.
4. EucDistance: Compute the pointwise Euclidean distance between two 2D fields.

Usage:
------
Import this script as a module or execute functions directly within a Python environment.
Ensure the required libraries are installed.
"""

import numpy as np
from eofs.xarray import Eof
import xarray as xr

# =============================================================================
# 1. Anomaly Calculation Function
# =============================================================================

def calculate_anomaly(da, groupby_type="time.dayofyear"):
    """
    Compute anomalies relative to the mean annual cycle (or another grouping).

    Parameters
    ----------
    da : xarray.DataArray
        Input data array with a 'time' dimension.
    groupby_type : str, optional
        By default, uses "time.dayofyear" for daily climatology. To compute monthly
        anomalies, use "time.month", etc.

    Returns
    -------
    xarray.DataArray
        The anomalies of 'da', computed as da - climatology.
    """
    gb = da.groupby(groupby_type)
    clim = gb.mean(dim="time")
    return gb - clim


# =============================================================================
# 2. EOF Analysis Function
# =============================================================================

def eofcalc_pcnorm(dat, w='sqrtcoslat', neofs=10, timeaxis='time', lonneg=None, latneg=None):
    """
    Compute EOF analysis on a DataArray (time, lat, lon), normalizing the PCs to unit variance.

    Parameters
    ----------
    dat : xarray.DataArray
        Data of shape (time, lat, lon). If the time dimension is not 'time',
        provide the dimension name via 'timeaxis'.
    w : str or xarray.DataArray, optional
        If 'sqrtcoslat', applies sqrt(cos(lat)) weighting. Otherwise, pass an array
        with the same shape as (lat, lon) for weights.
    neofs : int, optional
        Number of EOFs to compute (default=10).
    timeaxis : str, optional
        Name of the time dimension if not 'time'.
    lonneg : float, optional
        Longitude for flipping the EOF sign if the EOF at (lonneg, latneg) is > 0.
    latneg : float, optional
        Latitude for flipping the EOF sign if the EOF at (lonneg, latneg) is > 0.

    Returns
    -------
    pcs : xarray.DataArray
        Principal component time series (PCs) with unit variance, shape (time, neofs).
    eofs : xarray.DataArray
        EOF spatial patterns, shape (neofs, lat, lon).
    eofs_corr : xarray.DataArray
        EOFs as correlations, shape (neofs, lat, lon).
    var_fracs : xarray.DataArray
        Fraction of variance explained by each EOF (length=neofs).
    reconstruction : xarray.DataArray
        Data reconstructed from the specified number of EOFs (e.g., neofs).
    """
    #------------ Rename time axis if needed
    if timeaxis != 'time':
        dat = dat.rename({timeaxis: 'time'})

    #------------ Ensure the 'time' dimension is first
    if dat.dims[0] != 'time':
        dat = dat.transpose("time", ...)

    #------------ Compute weights
    if w == 'sqrtcoslat':
        #------------ sqrt(cos(lat)) weighting
        lat_radians = np.radians(dat.lat)
        weights = np.sqrt(np.cos(lat_radians))
        #------------ Expand dims to match (lat, lon)
        weights = weights.expand_dims(dim={'lon': dat.lon.size})
        weights = weights.transpose()
        weights['lon'] = dat.lon
    else:
        #------------ Assume 'w' is already a DataArray of shape (lat, lon)
        weights = w

    #------------ EOF solver
    solver = Eof(dat, weights=weights, center=True)

    #------------ Extract the number of EOFs requested
    pcs = solver.pcs(npcs=neofs, pcscaling=1)
    eofs = solver.eofs(neofs=neofs, eofscaling=2)
    eofs_corr = solver.eofsAsCorrelation(neofs=neofs)
    total_variance = solver.totalAnomalyVariance()

    #------------ Retrieve all variance fractions and slice up to neofs
    all_var_fracs = solver.varianceFraction()
    var_fracs = all_var_fracs[:neofs]

    #------------ Reconstruct the field using the specified number of EOFs
    reconstruction = solver.reconstructedField(neofs=neofs)
    #------------ Projection
    _ = solver.projectField(dat, neofs=3)  # pseudo_PCs

    # Optional: flip the sign of all EOFs & PCs if the value at (lonneg, latneg) is > 0
    if (lonneg is not None) and (latneg is not None):
        if (eofs.sel(lon=lonneg, lat=latneg, method='nearest') > 0).all():
            eofs = -1.0 * eofs
            pcs = -1.0 * pcs

    return pcs, eofs, eofs_corr, var_fracs, reconstruction


# =============================================================================
# 3. Projection Function
# =============================================================================

def proj_onto_eof(dat, eof, w='sqrtcoslat'):
    """
    Project a data array onto a pre-computed EOF pattern.

    Parameters
    ----------
    dat : xarray.DataArray
        Data to be projected, shape (time, lat, lon).
    eof : xarray.DataArray
        Pre-computed EOF pattern (lat, lon), or a single EOF.
    w : str or xarray.DataArray, optional
        If 'sqrtcoslat', uses sqrt(cos(lat)) weighting. Otherwise, pass an array
        matching (lat, lon).

    Returns
    -------
    xarray.DataArray
        Projection of 'dat' onto 'eof' (the resulting PC-like time series).
    """
    #------------ Handle weighting
    if w == 'sqrtcoslat':
        lat_radians = np.radians(dat.lat)
        weights = np.sqrt(np.cos(lat_radians))
        weights = weights.expand_dims(dim={'lon': dat.lon.size})
        weights = weights.transpose()
        weights['lon'] = dat.lon
    else:
        weights = w

    #------------ Numerator: dot product of (dat * weights) with the EOF
    num = (dat * weights).dot(eof, dims=["lat", "lon"])

    #------------ Denominator: dot product of EOF with itself
    denom = eof.dot(eof, dims=["lat", "lon"])

    return num / denom


# =============================================================================
# 4. Euclidean Distance Function
# =============================================================================
def EucDistance(field1, field2):
    """
    Compute the pointwise Euclidean distance between two 2D fields of shape (lat, lon).

    Since sqrt((x - y)^2) = abs(x - y), each grid cell returns the absolute difference.

    Parameters
    ----------
    field1 : np.ndarray or xarray.DataArray
        First 2D array (lat, lon) representing EOF spatial patterns.
    field2 : np.ndarray or xarray.DataArray
        Second 2D array (lat, lon) to compare against field1.

    Returns
    -------
    distance : np.ndarray or xarray.DataArray
        A 2D array of shape (lat, lon) containing the pointwise Euclidean distance
        (i.e., absolute difference) for each grid cell.
    """
    return np.abs(field1 - field2)
