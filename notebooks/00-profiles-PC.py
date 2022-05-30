# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np 
from pathlib import Path
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from astroquery.vizier import Vizier
import seaborn as sns

sns.set_context("talk")
sns.set_color_codes()

datapath = Path.cwd().parent / "data"

# # Get celestial coordinates of proplyd sources

# List of proplyds that we will work with:

proplyd_ids = [
    "177-341W",
    "180-331",
]

# ## Coordinates from SIMBAD (not accurate enough)

# Use SIMBAD name service to get the coordinates of each source

cdict = {
    _: SkyCoord.from_name(f"[RRS2008] {_}")
    for _ in proplyd_ids
}

cdict

# + tags=[]
cdict["177-341W"].to_string("hmsdms")
# -

# Even though we use the Ricci catalog ID to find the source, the SIMBAD coordinates _do not come from that catalog_. As a result, they are inaccurate by of order 0.2 arcsec, so we cannot use them

# ## Coordinates from Ricci 2008 catalog via Vizier

# Get the coordinates of all proplyds in the first table of the catalog. Add the `Name` column as an index, so we can more efficiently look up individual sources. 

proplyd_tab = Vizier(
    catalog="J/AJ/136/2136",
    columns=['Name', '_RAJ2000', '_DEJ2000'],
    row_limit=-1,
).get_catalogs(    
    catalog="J/AJ/136/2136",
)[0]
proplyd_tab.add_index("Name")

# Restrict to just the sources that we want:

proplyd_tab = proplyd_tab.loc[proplyd_ids]

# If we make minor modifications to the RA, Dec column names, then `SkyCoord.guess_from_table()` can do its magic. Then we can remove the original columns. 

proplyd_tab.rename_columns(['_RAJ2000', '_DEJ2000'], ["ra_J2000", "dec_J2000"])
proplyd_tab["ICRS"] = SkyCoord.guess_from_table(proplyd_tab)
proplyd_tab.remove_columns(["ra_J2000", "dec_J2000"])

proplyd_tab

# # Look at image of some sources

# + tags=[]
fname = "f631n"
hdu = fits.open(datapath / f"align-pcmos-{fname}_sharp_16.fits")[0]
# -

W = WCS(hdu.header)

source = "177-341W"
#source = "180-331"
x0, y0 = proplyd_tab.loc[source]["ICRS"].to_pixel(W)
fig, ax = plt.subplots(subplot_kw=dict(projection=W))
ax.imshow(hdu.data, vmin=0, vmax=2, cmap="gray_r")

# So, using imshow we always have the x, y axes aligned with the pixeaxes of the imegae. 
#
# If we want to have it aligned with equatorial axes, then we will have to define an auxiliary WCS and then use pcolormesh instead of imshow.
#
#

# + tags=[]
ny, nx = 100, 100
pixscale = Angle("0.05 arcsec").deg
c = proplyd_tab.loc[source]["ICRS"]
wc = WCS(naxis=2)
wc.wcs.cdelt = [-pixscale, pixscale]
wc.wcs.crval = [c.ra.deg, c.dec.deg]
wc.wcs.crpix = [0.5 * (1 + nx), 0.5 * (1 + ny)]
wc.wcs.ctype = ["RA---TAN", "DEC--TAN"]
# -

wc

# Now, we use upper case for the pixel coords and celestial coords of the big image grid

NY, NX = hdu.data.shape
X, Y = np.meshgrid(np.arange(NX), np.arange(NY))
CPIX = W.pixel_to_world(X, Y)

# So CPIX is an array of same shape as hdu.data that contains celestial coordinates.
#
# Check that if we convert them to pixels in our little aux frame that we have some near the origin

np.sum(np.hypot(*CPIX.to_pixel(wc)) < 50)

fig, ax = plt.subplots(
    figsize=(12, 12),
    subplot_kw=dict(projection=wc),
)
T = ax.get_transform("world")
ax.pcolormesh(
    CPIX.ra.deg,
    CPIX.dec.deg,
    hdu.data, 
    vmin=0, 
    vmax=15, 
    cmap="gray_r",
    shading="nearest",
    transform=T,
)
ax.scatter(c.ra.deg, c.dec.deg, transform=T, color='r')
ax.set_aspect("equal")
ax.set(
    xlim=[0, nx],
    ylim=[0, ny],
)

proplyd_tab.loc[source]["ICRS"].to_string("hmsdms")

proplyd_tab.loc[source]["ICRS"].to_string

# So this shows that we can put the images at arbitrary axes if we use `pcolormesh` bt that the coordinates are not good enough from the catalogs. We have made our own list of coordinates instead.

# # Read in list  of sources with my bespoke coordinates
#
# These are specialized to the case of the PC mosaic, since the alignment with Robberto is still not perfect, even though I have done my best.

import regions

regfile = datapath / "pcmos-proplyds.reg"
regs = regions.Regions.read(datapath / regfile, format="ds9")

source_list = [{"Name": r.meta["label"],  "ICRS": r.center} for r in regs]

import pandas as pd
from astropy.table import Table

source_table = Table(source_list)
source_table

# ## Find PA to th1C

c0 = SkyCoord.from_name("* tet01 Ori C")

source_table["PA"] = source_table["ICRS"].position_angle(c0).to(u.deg)
source_table["Sep"] = source_table["ICRS"].separation(c0).to(u.arcsec)
source_table.add_index("Name")

source_table

source_table.loc[source]

# + tags=[]
ny, nx = 100, 100
pixscale = Angle("0.05 arcsec").deg
c = source_table.loc[source]["ICRS"]
wc = WCS(naxis=2)
wc.wcs.cdelt = [-pixscale, pixscale]
wc.wcs.crval = [c.ra.deg, c.dec.deg]
wc.wcs.crpix = [0.5 * (1 + nx), 0.5 * (1 + ny)]
wc.wcs.ctype = ["RA---TAN", "DEC--TAN"]
# -

fig, ax = plt.subplots(
    figsize=(12, 12),
    subplot_kw=dict(projection=wc),
)
T = ax.get_transform("world")
ax.pcolormesh(
    CPIX.ra.deg,
    CPIX.dec.deg,
    hdu.data, 
    vmin=0, 
    vmax=15, 
    cmap="gray_r",
    shading="nearest",
    transform=T,
)
ax.scatter(c.ra.deg, c.dec.deg, transform=T, color='r')
ax.set_aspect("equal")
ax.set(
    xlim=[0, nx],
    ylim=[0, ny],
)

# +
# ax.quiver?
# -


