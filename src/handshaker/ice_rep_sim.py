"""Basic toolkit to implement model of ice transmission that is
representative of that due to water ice growth on the FPA.

A representative (not realistic) ice growth rate map is loaded from
rate_mosaic_filename below.  Those rates are assumed to be constant
over time. i.e. a steady ice growth rate.  A simplistic decon schedule
is also set up so that at certain times from epoch zero
(cadence_start_date below), the ice thickness is reset to zero over
the entire FPA.  The ice decon cadence is set by decon_period below,
which is set to 20 days.  Then, based on a user-supplied FPA position
and MJD, an ice thickness is computed.  The MJD supplied must be no
ealier than epoch 0 (MJD 61295.0 or 2026-09-12T00:00:00.0).  Next, the
transmission info in interim_T_filenames is used to compute a
tansmission-vs-wavelength curve relative to the reference epoch.  That
curve is currently simply an interpolation of the ice-thickness
dependent transmission curves near the middle of WFI10 inferred from
SCIPA TVAC data.

All lengths (wavelength, ice thickness) are in nm.  All times are in
days.

9/21/2026

"""

import numpy as np
from astropy.time import Time
from astropy.table import Table

cadence_start_date = Time('2026-09-12T00:00:00.0', format='isot', scale='utc')
decon_period = 20 # days

rate_mosaic_filename = 'ice_rate_mosaic.npz'
interim_T_filename = 'spectral_responses_SCA10.ecsv'

def ice_growth_rate(sca, pix_row_col):

    """Returns toy-model ice growth rate given an SCA number and
    position within it in pixel coordinates. i.e. x and y both in
    the 0-4096 range.

    Parameters
    ----------

    sca : int
      1-indexed FPA detector number (e.g. 2 for WFI02)
    pix_row_col : tuple
      the (y, x) position within the detector in pixel coordinates
    
    Returns
    -------

    The model ice growth rate in nm/day

    """

    dict = np.load(rate_mosaic_filename, allow_pickle=True)

    rate_mosaic = dict['rate_mosaic']
    det_row_offsets = dict['det_row_offsets'].item()
    det_col_offsets = dict['det_col_offsets'].item()
    pix_per_bin = dict['pix_per_bin'].item()

    yind = pix_row_col[0]//pix_per_bin + det_row_offsets[sca]
    xind = pix_row_col[1]//pix_per_bin + det_col_offsets[sca]

    return rate_mosaic[yind, xind]


def ice_thickness(sca, pix_row_col, time_mjd):

    """Returns toy-model ice thickness given an SCA number and
    position within it in pixel coordinate. i.e. x and y both in
    the 0-4096 range.

    Parameters
    ----------

    sca : int
      1-indexed FPA detector number (e.g. 2 for WFI02)
    pix_row_col : tuple
      the (y, x) position within the detector in pixel coordinates
    time_mjd : float
      MJD for which ice transmission is desired; must be no ealier
      than   The MJDs used must be no earlier than
      61295.0
    
    Returns
    -------

    The model ice thinkness in nm

    """

    time_since_decon = (time_mjd - cadence_start_date.mjd) % decon_period

    return ice_growth_rate(sca, pix_row_col)*time_since_decon


def ice_relative_transmission(sca, pix_row_col, time_mjd):

    """Returns toy-model ice transmission relative to epoch 0 given an
    SCA number and position within it in pixel coordinate. i.e. x and
    y both in the 0-4096 range.

    Parameters
    ----------

    sca : int
      1-indexed FPA detector number (e.g. 2 for WFI02)
    pix_row_col : tuple
      the (y, x) position within the detector in pixel coordinates
    time_mjd : float
      MJD for which ice transmission is desired; must be no ealier
      than   The MJDs used must be no earlier than
      61295.0
    
    Returns
    -------

    Two arrays: first is wavelength in nm; the second is transmission
    as a fraction relative to the transmission at epoch 0.

    and

    a scalar (float) which gives the ice layer thickness in nm 
    
    """

    tab = Table.read(interim_T_filename)
    colnames = tab.colnames
    tab_thicknesses = np.array([float(s[s.find('_d')+2:s.find('nm')]) for s in colnames if '_d' in s])
    want_thickness = ice_thickness(sca, pix_row_col, time_mjd)
    right_edgenum = np.digitize(want_thickness, tab_thicknesses)
    neighbor_tab_thicknesses = tab_thicknesses[right_edgenum-1:right_edgenum+1]
    # linear interpolation weights between neighboring passbands
    wt = np.abs(want_thickness - neighbor_tab_thicknesses) / np.diff(neighbor_tab_thicknesses)[0]
    passband = wt[1]*tab[tab.colnames[right_edgenum]] + wt[0]*tab[tab.colnames[right_edgenum+1]]

    return tab['wavelength'], passband, want_thickness
