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
from astropy.nddata import Cutout2D
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
source_list = [
    {"Name": r.meta["label"],  "ICRS": r.center} 
    for r in regs 
    if "label" in r.meta
]
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

# ## Cutout image of a proplyd in a particular filter

# + [markdown] tags=[]
# We use `astropy.nddata.Cutout2D` to make cutouts of the sharp image, the smooth image, and the coordinate array. 
# -

class ProplydCutout:
    
    def __init__(self, pdata, image: FilterImage, size=2 * u.arcsec):
        self.center = pdata["ICRS"]
        self.pa_star = pdata["PA"]
        self.sep = pdata["Sep"]
        self.pname = pdata["Name"]
        self.fname = image.name
        self.size = size
        self.cutout = Cutout2D(
            image.data, position=self.center, size=size, wcs=image.wcs, copy=True,
        )
        self.image = self.cutout.data
        self.wcs = self.cutout.wcs
        # Use the slices from this cutout to also get cutout of the smoothed data array
        self.smooth_image = image.sdata[self.cutout.slices_original]
        # ... and the same for the coordinates
        self.image_coords = image.coords[self.cutout.slices_original]
        # Radius and PA of each pixel with respect to the center
        self.r = self.center.separation(self.image_coords)
        self.pa = self.center.position_angle(self.image_coords)
        # Default mask has max radius of half the cutout size
        self.set_mask(r_out=self.size / 2)
        self.owcs = self.get_ortho_wcs()
        
    def __repr__(self):
        return f"ProplydCutout({self.pname}, {self.fname})"
       
    def get_ortho_wcs(self):
        """Auxilary WCS that is orthogonal to RA, Dec
        
        Pixel size is 1 arcsec. 
        Origin is set at corner pixel, 
        which is (1, 1) in FITS, but is (0, 0) in python
        """
        wcs = WCS(naxis=2)
        wcs.wcs.cdelt = [-1/3600, 1/3600]
        wcs.wcs.crval = [self.center.ra.deg, self.center.dec.deg]
        wcs.wcs.crpix = [1, 1]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return wcs
         
    def set_mask(
        self,
        r_out = 1.0 * u.arcsec,
        r_in = 0.1 * u.arcsec,
        mu_min = -0.0,
    ):
        cth = np.cos((self.pa - self.pa_star))
        self.mask = (self.r <= r_out) & ((cth >= mu_min) | (self.r <= r_in))


# Test that the cutout works:

p = ProplydCutout(source_table.loc["177-341W"], imdict["f547m"])

# We can plot it with imshow, but that is rotated with respect to equatorial axes.

fig, ax = plt.subplots(subplot_kw=dict(projection=p.wcs))
ax.imshow(p.image, vmin=0, vmax=15, cmap="gray_r")
ax.scatter(
    p.center.ra.deg, 
    p.center.dec.deg, 
    transform=ax.get_transform("world"), 
    color='r', 
    marker="+",
    s=50,
)
...;

# We can use the orthogonal wcs and pcolormesh to rotate the image so that axes are aligned with RA and Dec.

fig, ax = plt.subplots(
    figsize=(6, 6),
    subplot_kw=dict(projection=p.owcs),
)
T = ax.get_transform("world")
ax.pcolormesh(
    p.image_coords.ra.deg,
    p.image_coords.dec.deg,
    np.where(p.mask, p.image, np.nan), 
    vmin=0, 
    vmax=10, 
    cmap="gray_r",
    shading="nearest",
    transform=T,
)
ax.scatter(
    p.center.ra.deg, 
    p.center.dec.deg, 
    transform=T, 
    color='r', 
    marker="+",
    s=300,
)
ax.set_aspect("equal")
ax.set(
    xlim=[-1, 1],
    ylim=[-1, 1],
)
...;

j, i = np.unravel_index(np.argmin(p.r, axis=None), p.r.shape)
p.r[j, i].arcsec

# ## Make cutout for all proplyds and all filters
#
# Add them in to the table of sources

for fname in pcfilters:
    source_table[fname] = [ProplydCutout(row, imdict[fname]) for row in source_table]

source_table[pcfilters]

# ## Do the images

# + tags=[]
nprops = len(source_table)
ns = len(pcfilters)
fig, axes = plt.subplots(nprops, ns, figsize=(3 * ns, 3 * nprops))
for j, row in enumerate(source_table):
    for i, fname in enumerate(pcfilters):
        ax = axes[j, i]
        ax.imshow(row[fname].image**0.5, vmin=0.8, vmax=10, cmap="magma_r", origin='lower')
        ax.text(0.05, 0.95, fname.upper(), transform=ax.transAxes, va="top", ha="left")
        ax.text(0.95, 0.05, row["Name"], transform=ax.transAxes, va="bottom", ha="right")
        p = row[fname]
        xp, yp = p.center.to_pixel(p.wcs)
        ax.scatter(xp, yp, color="w", marker="+", s=50)
        j0, i0 = np.unravel_index(np.argmin(p.r, axis=None), p.r.shape)
        #ax.scatter(i0, j0, color="g", marker="x", s=50)
        ax.set(xticks=[], yticks=[])
sns.despine(left=True, bottom=True)
fig.tight_layout(pad=0, h_pad=0.1, w_pad=0.1)

# + [markdown] tags=[]
# ## Do the profiles
# -

nbins = 20
fig, axes = plt.subplots(nprops, ns, figsize=(3 * ns, 2.0 * nprops), sharex=True, sharey='row')
for j, row in enumerate(source_table):
    for i, fname in enumerate(pcfilters):
        p = row[fname]
        ax = axes[j, i]
        m = p.mask
        ax.scatter(
            p.r.arcsec[m], p.image[m],
            c=(p.pa - p.pa_star)[m],
            alpha=0.3,
            cmap="magma_r",
            s=20,
        )
        h1, edges = np.histogram(p.r.arcsec[m], range=[0.0, 1.0], bins=nbins, weights=p.image[m])
        h0, edges = np.histogram(p.r.arcsec[m], range=[0.0, 1.0], bins=nbins)
        rgrid = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(rgrid, h1 / h0, drawstyle="steps-mid", linewidth=3)
        ax.text(1.0, 0.8, fname.upper(), transform=ax.transAxes, va="top", ha="right")
        ax.text(1.0, 1.0, row["Name"], transform=ax.transAxes, va="top", ha="right")
        ax.set(ylim=[0.0, None], xlim=[0.0, None])
sns.despine()
fig.tight_layout(h_pad=0.3, w_pad=0.3)

# I have made sure that all the profiles in the same row share a common y scale, so that we can easily compare the different profiles. 
#
# We can see that in many cases the 631 profile is significantly higher than the sum of the 547 and the 656.  This is good evidence that we are seeing [O I] from the neutral disk wind. 
#
# Not all sources show this however. For instance, 167-317 is pretty dominated by Ha.
#
# We have also added a spatial averaging of the profiles, calculated using a weighted histogram. This is devloped further in the following section. 

# ## Put all the profiles on a common uniform grid of radii

# Test method for determining bin edges that is robust to rounding errors:

# +
bin_size = 0.05
epsilon = 0.01 # Fake rounding error
rmin = 0.0 + epsilon
rmax = 1.0 - epsilon

nbins = int(np.round((rmax - rmin) / bin_size))
edges = np.arange(nbins + 1) * bin_size
edges


# +
class ProplydProfiles:
    """
    Radial profiles of proplyd brightness for multiple filters
    
    All interpolated onto a common grid of radial points
    """
    bin_size = 0.05
    
    def __init__(self, data, fnames):
        # Setup the radial bins
        rmax = data["f631n"].size.to(u.arcsec).value / 2
        rmin = 0.0
        nbins = int(np.round((rmax - rmin) / bin_size))
        edges = np.arange(nbins + 1) * bin_size
        # Convert to cell centers
        self.r = 0.5 * (edges[1:] + edges[:-1])
        # Initialize Tables to hold results
        self.mean = Table({"r": self.r})
        self.sigma = Table({"r": self.r})
        self.npix = Table({"r": self.r})
        # Copy some metadata from the proplyd
        self.name = data["Name"]
        # Initialise dicts to hold the smooth BG mean and sigma
        self.bgmean = {}
        self.bgsig = {}
        # Process each filter
        for fname in fnames:
            p = data[fname]
            m = p.mask
            # Use weighted histograms to get mean and stddev
            h1, _ = np.histogram(
                p.r.arcsec[m], 
                range=[rmin, rmax], 
                bins=edges, 
                weights=p.image[m],
            )
            h0, _ = np.histogram(
                p.r.arcsec[m], 
                range=[rmin, rmax], 
                bins=edges, 
                weights=None,
            )
            h2, _ = np.histogram(
                p.r.arcsec[m], 
                range=[rmin, rmax], 
                bins=edges, 
                weights=p.image[m]**2,
            )
            # Number of image pixels that contribute to each radial bin
            self.npix[fname] = h0
            # Mean brightness in each radial bin
            self.mean[fname] = h1 / h0
            # Standard deviation of brightness in each radial bin
            self.sigma[fname] = np.sqrt((h2 / h0) - (h1 / h0)**2)
            # Also save smooth BG values
            self.bgmean[fname] = np.mean(p.smooth_image)
            self.bgsig[fname] = np.std(p.smooth_image)            

            

        

        

# + tags=[]
pp = ProplydProfiles(source_table.loc["177-341W"], pcfilters)
# -

pp.mean

pp.npix

# Check that the BG mean and sigma are working.  Note that these still need to be flux callibrated

# + tags=[]
pp.bgmean
# -

pp.bgsig

# The sigma is very small (5 to 10%) because we are sampling a region that is smaller than the smoothing scale, so we might as well ignore it. 

fig, ax = plt.subplots()
for fname in pcfilters:
    line, = ax.plot("r", fname, data=pp.mean, label=fname.upper())
    ax.fill_between(
        pp.mean["r"], 
        pp.mean[fname] - 2 * pp.sigma[fname] / np.sqrt(pp.npix[fname]), 
        pp.mean[fname] + 2 * pp.sigma[fname] / np.sqrt(pp.npix[fname]), 
        color=line.get_color(),
        alpha=0.1,
        linewidth=0,
    )
ax.legend()
ax.set_title(pp.name)
ax.set(
    xlabel="Radius, arcsec",
    ylabel="Brightness",
)
...;

# ## Isolate the 6300 emission from the neutral flow

# We want to subtract two things from the raw F631N brightness:
#
# 1. The continuum emission in the filter bandpass. This is mainly direct starlight, but will include a bit of scattered starlight and bound-free atomic continuum too. 
#
# 2. The [O I] 6300 line emission that comes from the ionization front instead of from the neutral gas. This should be sharply peaked at an ionization fraction of $x = 0.5$ because of the $x (1 - x)$ dependence.

# For (1) we can subtract off the broader band filter F547M.  But if we only want to subtract the star part, then we can use a scaled version of Ha F656N to estimate the atomic continuum contribution. I choose a value of `atfac` so that the emission at the i-front is cancelled out. Note that this ignores the fact that F656N itself will have a small continuum contribution. *This does not work very well – I am going to try to calculate it better*
#
# For (2), we can just subtract the Ha F656N profile. This will over-correct for fully ionized part of the proplyd flow, which might lead to negative parts of the profile. 

# ### Pilot study of 177-341W

# +
atfac = 0.55
cont = pp.mean["f547m"] - atfac * (pp.mean["f656n"] - 1)
oin = pp.mean["f631n"] - (cont - 1) - (pp.mean["f656n"] - 1)
sig = np.sqrt(
    pp.sigma["f547m"]**2 / pp.npix["f547m"] 
    + pp.sigma["f631n"]**2 / pp.npix["f631n"] 
    + pp.sigma["f656n"]**2 / pp.npix["f656n"]
)


fig, ax = plt.subplots()
line, = ax.plot(pp.mean["r"], oin - 1, label="residual oi", linewidth=5)
ax.fill_between(
    pp.mean["r"], 
    (oin - 1) - 3 * sig, 
    (oin - 1) + 3 * sig, 
    color=line.get_color(),
    alpha=0.1,
    linewidth=0,
)
ax.plot(pp.mean["r"], pp.mean["f656n"] - 1, label="ha")
ax.plot(pp.mean["r"], cont - 1, label="starlight")
ax.plot(pp.mean["r"], pp.mean["f631n"] - 1, label="oi + cont")

ax.axhline(0.0, linestyle="dashed", color="k")
ax.legend()
ax.set_title(pp.name)
ax.set(
    xlabel="Radius, arcsec",
    ylabel="Brightness",
)
...;


# -

# ### Extracting the profile of stellar continuum
#
# What I will try and do is to assume that beyond a certain point (say 0.2 arcsec) all the continuum is atomic and therefore proportional to Ha.  So I will calculate the average F547M/F656N in that range and use it as my `atfac` to help isolate the stellar peak at the origin.

def find_atomic_factor(pp: ProplydProfiles, rstar=0.2):
    m = pp.r >= rstar
    f547m = np.sum(pp.mean["f547m"][m] - 1)
    f656n = np.sum(pp.mean["f656n"][m] - 1)
    return f547m / f656n


pp = ProplydProfiles(source_table.loc["180-331"], pcfilters)
find_atomic_factor(pp)

# For some of the sources, this does not give a reasonable value (which I am defining as between 0.2 and 1.0), so for those I will just assume 0.5.

# ### Isolated 6300 profile for all sources

# + tags=[]
weak_sources = [
    "156-308 NEW", "174-414", "183-427", "183-419", "182-413", 
    "175-355", "152-319", "154-321 NEW", "154-324", "179-354",
]
bright_sources = ["167-317", "163-317", "158-323", "171-340", "176-325"]


# +
class ProplydResults:
    """Final results extracted from filter images"""
    def __init__(self, pp: ProplydProfiles):
        # Left edge of bins to improve plotting
        self.rleft = pp.r - pp.bin_size / 2
        self.pp = pp
        
        # Total profile in F631N filter
        self.f631n = pp.mean["f631n"]
        # Ignore potential contamination of Ha filter for now
        self.ha = pp.mean["f656n"]
        # Profiles of stellar continuum: subtract atomic cont from F547M
        self.atfac = find_atomic_factor(pp)
        if not (0.2 < self.atfac < 1.0):
            self.atfac = 0.5
        self.cont = pp.mean["f547m"] - self.atfac * (self.ha - 1)
        # Do not let it go below BG 
        self.cont = np.maximum(self.cont, 1.0)
        # [O I] 6300 emission with continuum and i-front removed
        self.oin = self.f631n - (self.cont - 1) - (self.ha - 1)
        # Assume all the errors can be added in quadrature
        self.sig = np.sqrt(
            pp.sigma["f547m"]**2 / pp.npix["f547m"] 
            + pp.sigma["f631n"]**2 / pp.npix["f631n"] 
            + pp.sigma["f656n"]**2 / pp.npix["f656n"]
        )
         


# +
ncols = 3
nrows = nprops // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))

for ax, row in zip(axes.flat, source_table):
    pp = ProplydProfiles(row, pcfilters)
    pres = ProplydResults(pp)
    
    oiline, = ax.plot(
        pres.rleft, pres.oin - 1, 
        label="residual [O I]", linewidth=5,
        drawstyle="steps-post",
    )
    ax.fill_between(
        pres.rleft, 
        (pres.oin - 1) - 1 * pres.sig, 
        (pres.oin - 1) + 1 * pres.sig, 
        color=oiline.get_color(),
        alpha=0.3,
        linewidth=0,
        step="post",
    )
    haline, = ax.plot(
        pres.rleft, pres.ha - 1, 
        label="Hα",
        drawstyle="steps-post",
    )
    cline, = ax.plot(
        pres.rleft, pres.cont - 1, 
        label="starlight",         
        drawstyle="steps-post",
        color="g",
        linestyle="dashed",
        linewidth=3,
    )
    ax.fill_between(
        pres.rleft, 
        0.0, 
        pres.cont - 1, 
        color=cline.get_color(),
        alpha=0.1,
        linewidth=0,
        step="post",
    )
    oiline2, = ax.plot(
        pres.rleft, pres.f631n - 1, 
        label="F631N",         
        drawstyle="steps-post",
        color=oiline.get_color(),
        linewidth=1,
    )


    #ax.plot(pp.mean["r"], pp.mean["f631n"] - 1, label="oi + cont")

    ax.axhline(0.0, linewidth=1, color="k")
    
    #ax.legend()
    label = row['Name']
    label += "\n" + f"[O I] BG = {pp.bgmean['f631n']:.2f}"
    label += "\n" + f"Hα BG = {pp.bgmean['f656n']:.2f}"

    ax.text(1.0, 1.0, label, transform=ax.transAxes, va="top", ha="right")
    if row['Name'] in weak_sources:
        yscale = 2.0
    elif row['Name'] in bright_sources:
        yscale = 25.0
    else:
        yscale = 12.0
    ax.set(ylim=[-yscale / 5, yscale])


for ax in axes[-1, :]:
    ax.set(xlabel="Radius, arcsec")
for ax in axes[:, 0]:
    ax.set(ylabel="Brightness / BG")

fig.legend(
    handles=[oiline2, oiline, haline, cline], 
    ncol=4, 
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
)

sns.despine()
fig.tight_layout()
...;
# -



# + tags=[]

