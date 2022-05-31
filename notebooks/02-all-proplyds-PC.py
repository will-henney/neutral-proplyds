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

# # Extract profiles for all proplyds and all filters

# ## Library imports

# ### General libraries

import numpy as np 
from pathlib import Path
import pandas as pd

# ### Astronomy libraries

from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, QTable
import regions

# ### Graphics libraries

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_context("talk")
sns.set_color_codes()

# ## Paths to the data files

datapath = Path.cwd().parent / "data"

# ## Get all the images we need in various filters

pcfilters = ["f631n", "f673n", "f656n", "f658n", "f547m", ]


class FilterImage:
    """WFPC2 PC image in a given filter
    
    Includes the following attributes:
    
    name: name of filter
    data: pixel image (hi-pass filtered)
    sdata: pixel image (lo-pass filtered)
    coords: celestial coordinates of pixels (same shape as data)
    wcs: an astropy.wcs.WCS instance
    """
    def __init__(self, name):
        self.name = name
        hdu = fits.open(
            datapath / f"align-pcmos-{self.name}_sharp_16.fits"
        )[0]
        self.wcs = WCS(hdu.header)
        self.data = hdu.data
        self.sdata = fits.open(
            datapath / f"pcmos-{self.name}_smooth_16.fits"
        )[0].data
        ny, nx = self.data.shape
        self.coords = self.wcs.pixel_to_world(
            *np.meshgrid(np.arange(nx), np.arange(ny))
        )
        


imdict = {
    name: FilterImage(name) for name in pcfilters
}

# ## Get all the proplyd source coordinates

# Get the sources from the DS9 region file that I made by hand.

# + tags=[]
regfile = datapath / "pcmos-proplyds.reg"
regs = regions.Regions.read(regfile, format="ds9")
# -

# Use θ¹ Ori C as origin to define position angles.

# + tags=[]
c0 = SkyCoord.from_name("* tet01 Ori C")
# -

# Extract the information that we want from the region file. Construct a list of dicts that give source name and coordinates:

# + tags=[]
source_list = [{"Name": r.meta["label"],  "ICRS": r.center} for r in regs]
# -

# Convert to an `astropy.table.QTable` of the sources. Add columns for PA to θ¹ Ori C (in degrees) and Separation from θ¹ Ori C (in arcsec):

# + tags=[]
source_table = QTable(source_list)
source_table["PA"] = source_table["ICRS"].position_angle(c0).to(u.deg)
source_table["Sep"] = source_table["ICRS"].separation(c0).to(u.arcsec)
source_table.add_index("Name")
source_table
# -

# By turning the `Name` column into an index, we can extract a given source by name. For example:

# + tags=[]
source_table.loc["182-413"]
# -

# The advantage of using a `QTable` is that the units remain attached to the values:

# + tags=[]
source_table.loc["177-341W"]["PA"], source_table.loc["180-331"]["Sep"]


# -

# ## Class to represent combination of a proplyd and an image in a particular filter

class ProplydImage:
    
    def __init__(self, pdata, image: FilterImage):
        self.c = pdata["ICRS"]
        self.pa = pdata["PA"]
        self.sep = pdata["Sep"]
        self.image = image

    def set_mask(
        self,
        r_out = 1.0 * u.arcsec,
        r_in = 0.1 * u.arcsec,
        mu_min = -0.2,
    ):
        r = self.c.separation(self.image.coords)
        pa = self.c.position_angle(self.image.coords)
        cth = np.cos((pa - self.pa))
        self.mask = (r <= r_out) & ((cth >= mu_min) | (r <= r_in))


# ## Do the profiles

source_data = source_table.loc["177-341W"]


# ## Do the images


