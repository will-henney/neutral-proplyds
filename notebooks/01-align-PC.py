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

# # Match PC coordinates to Robberto frame

from pathlib import Path
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u

datapath = Path.cwd().parent / "data" 
fname1 = "hst-acs-f658n-trap-south-2005_BGSUB"
fname2 = "pcmos-f631n_sharp_16"
sname1 = "fwhm4.0-thresh20"
sname2 = "fwhm4.0-thresh5"

hdu1, = fits.open(datapath / f"{fname1}.fits")
hdu2, = fits.open(datapath / f"{fname2}.fits")
w1 = WCS(hdu1.header)
w2 = WCS(hdu2.header)
stab1 = Table.read(datapath / f"{fname1}-sources-{sname1}.ecsv")
stab2 = Table.read(datapath / f"{fname2}-sources-{sname2}.ecsv")

c1 = w1.pixel_to_world(stab1["xcentroid"], stab1["ycentroid"])
c2 = w2.pixel_to_world(stab2["xcentroid"], stab2["ycentroid"])

max_sep = 1.0 * u.arcsec
idx, d2d, d3d = c1.match_to_catalog_sky(c2)
sep_constraint = d2d < max_sep
c1_matches = c1[sep_constraint]
c2_matches = c2[idx[sep_constraint]]
f"Out of {len(c1)} sources in catalog #1 we have {len(c1_matches)} in catalog #2, but only {len(set(idx[sep_constraint]))} are unique."

stab1_matches = stab1[sep_constraint]
stab2_matches = stab2[idx[sep_constraint]]


def match_catalogs(c1, c2, max_sep=1.0 * u.arcsec):
    """Find indices of coincident sources between two lists of sky coordinates
    
    Returns a pair of index arrays of the matched sources in each catalog.
    Each source is guaranteed to appear only once.
    """
    # Find closest source in c2 to each source in c1
    idx2, d2d, d3d = c1.match_to_catalog_sky(c2)
    # Make index of sources in c1: [0, 1, 2, ...]
    idx1 = np.arange(len(c1))
    # Mask that is True when closest source is close enough for match
    isclose = d2d < max_sep
    # Remove duplicate sources in the idx2 list by making a BACKWARDS mapping from 2 -> 1
    # This will keep only the last source in c2 that was matched by a source in c1
    backmap = dict(zip(idx2[isclose], idx1[isclose]))
    # Retrieve the two matched index lists from the dict
    imatch2, imatch1 = zip(*backmap.items())
    # Return arrays of matched indices, which can be used to obtain the matched sources
    return np.asarray(imatch1), np.asarray(imatch2)


ii1, ii2 = match_catalogs(c1, c2, max_sep=max_sep)

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("talk")

c1m = c1[ii1]
c2m = c2[ii2]
seps = c1m.separation(c2m)
np.alltrue(seps <= max_sep), seps.arcsec

fig, ax = plt.subplots()
ax.hist(seps.arcsec, bins=20)
ax.set(xlabel="separation, arcsec")
...;

# +
offsets = c1m.spherical_offsets_to(c2m)
ra, dec = [_.milliarcsecond for _ in offsets]
max_sep_mas = max_sep.to(u.milliarcsecond).value
limits = [-max_sep_mas, max_sep_mas]
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(ra, dec, alpha=0.25, linewidths=0)
ax.axvline(0.0, color="k", lw=3, linestyle="dashed")
ax.axhline(0.0, color="k", lw=3, linestyle="dashed")
ax.set(
    xlim=limits,
    ylim=limits,
    xlabel="displacement RA, milliarcsec", ylabel="displacement DEC, milliarcsec",
)

ax.set_aspect("equal")
...;
# -

fig, ax = plt.subplots(figsize=(12, 4))
ax.hist([ra, dec], bins=31, label=["ra", "dec"])
ax.set(xlabel="separation, milliarcsec", yscale="linear", xlim=limits)
ax.axvline(0.0, color="k", lw=3, linestyle="dashed")
ax.legend()
...;

from astropy.stats import mad_std, sigma_clipped_stats
ra_mean, ra_median, ra_std = sigma_clipped_stats(ra, sigma=2.0)
dec_mean, dec_median, dec_std = sigma_clipped_stats(dec, sigma=2.0)

# +
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(ra, dec, alpha=0.3, linewidths=0, label="matched sources")
ax.scatter(ra_median, dec_median, marker="x", s=400, label="sig-clip median")
ax.scatter(ra_mean, dec_mean, marker="x", s=400, label="sig-clip mean")
ax.errorbar(ra_mean, dec_mean, xerr=ra_std, yerr=dec_std, color="k", capsize=6, capthick=2, alpha=0.3, label="sig-clip stdev")
ax.axvline(0.0, color="k", lw=3, linestyle="dashed")
ax.axhline(0.0, color="k", lw=3, linestyle="dashed")
ax.legend(ncol=2)
ax.set(
    xlim=[-500, 200],
    ylim=[-500, 200],
    xlabel="displacement RA, milliarcsec", ylabel="displacement DEC, milliarcsec",
)

ax.set_aspect("equal")
# -

Angle(ra_median * u.milliarcsecond).deg, Angle(dec_median * u.milliarcsecond).deg

hdu2.header["CRVAL1"] -= Angle(ra_median * u.milliarcsecond).deg
hdu2.header["CRVAL2"] -= Angle(dec_median * u.milliarcsecond).deg

newname2 = f"align-{fname2}"
hdu2.writeto(datapath / f"{newname2}.fits", overwrite=True)



# ## Repeat for F547M

fname2 = "pcmos-f547m_sharp_16"
hdu2, = fits.open(datapath / f"{fname2}.fits")
w2 = WCS(hdu2.header)
stab2 = Table.read(datapath / f"{fname2}-sources-{sname2}.ecsv")
c2 = w2.pixel_to_world(stab2["xcentroid"], stab2["ycentroid"])

max_sep = 1.0 * u.arcsec
idx, d2d, d3d = c1.match_to_catalog_sky(c2)
sep_constraint = d2d < max_sep
c1_matches = c1[sep_constraint]
c2_matches = c2[idx[sep_constraint]]
f"Out of {len(c1)} sources in catalog #1 we have {len(c1_matches)} in catalog #2, but only {len(set(idx[sep_constraint]))} are unique."

stab1_matches = stab1[sep_constraint]
stab2_matches = stab2[idx[sep_constraint]]
ii1, ii2 = match_catalogs(c1, c2, max_sep=max_sep)
c1m = c1[ii1]
c2m = c2[ii2]
seps = c1m.separation(c2m)

offsets = c1m.spherical_offsets_to(c2m)
ra, dec = [_.milliarcsecond for _ in offsets]
max_sep_mas = max_sep.to(u.milliarcsecond).value
limits = [-max_sep_mas, max_sep_mas]

fig, ax = plt.subplots(figsize=(12, 4))
ax.hist([ra, dec], bins=31, label=["ra", "dec"])
ax.set(xlabel="separation, milliarcsec", yscale="linear", xlim=limits)
ax.axvline(0.0, color="k", lw=3, linestyle="dashed")
ax.legend()
...;

ra_mean, ra_median, ra_std = sigma_clipped_stats(ra, sigma=2.0)
dec_mean, dec_median, dec_std = sigma_clipped_stats(dec, sigma=2.0)

# +
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(ra, dec, alpha=0.3, linewidths=0, label="matched sources")
ax.scatter(ra_median, dec_median, marker="x", s=400, label="sig-clip median")
ax.scatter(ra_mean, dec_mean, marker="x", s=400, label="sig-clip mean")
ax.errorbar(ra_mean, dec_mean, xerr=ra_std, yerr=dec_std, color="k", capsize=6, capthick=2, alpha=0.3, label="sig-clip stdev")
ax.axvline(0.0, color="k", lw=3, linestyle="dashed")
ax.axhline(0.0, color="k", lw=3, linestyle="dashed")
ax.legend(ncol=2)
ax.set(
    xlim=[-500, 200],
    ylim=[-500, 200],
    xlabel="displacement RA, milliarcsec", ylabel="displacement DEC, milliarcsec",
)

ax.set_aspect("equal")
# -

hdu2.header["CRVAL1"] -= Angle(ra_median * u.milliarcsecond).deg
hdu2.header["CRVAL2"] -= Angle(dec_median * u.milliarcsecond).deg
newname2 = f"align-{fname2}"
hdu2.writeto(datapath / f"{newname2}.fits", overwrite=True)

# ## Repeat for F673N

fname2 = "pcmos-f673n_sharp_16"
hdu2, = fits.open(datapath / f"{fname2}.fits")
w2 = WCS(hdu2.header)
stab2 = Table.read(datapath / f"{fname2}-sources-{sname2}.ecsv")
c2 = w2.pixel_to_world(stab2["xcentroid"], stab2["ycentroid"])

max_sep = 1.0 * u.arcsec
idx, d2d, d3d = c1.match_to_catalog_sky(c2)
sep_constraint = d2d < max_sep
c1_matches = c1[sep_constraint]
c2_matches = c2[idx[sep_constraint]]
f"Out of {len(c1)} sources in catalog #1 we have {len(c1_matches)} in catalog #2, but only {len(set(idx[sep_constraint]))} are unique."

stab1_matches = stab1[sep_constraint]
stab2_matches = stab2[idx[sep_constraint]]
ii1, ii2 = match_catalogs(c1, c2, max_sep=max_sep)
c1m = c1[ii1]
c2m = c2[ii2]
seps = c1m.separation(c2m)

offsets = c1m.spherical_offsets_to(c2m)
ra, dec = [_.milliarcsecond for _ in offsets]
max_sep_mas = max_sep.to(u.milliarcsecond).value
limits = [-max_sep_mas, max_sep_mas]

fig, ax = plt.subplots(figsize=(12, 4))
ax.hist([ra, dec], bins=31, label=["ra", "dec"])
ax.set(xlabel="separation, milliarcsec", yscale="linear", xlim=limits)
ax.axvline(0.0, color="k", lw=3, linestyle="dashed")
ax.legend()
...;

ra_mean, ra_median, ra_std = sigma_clipped_stats(ra, sigma=2.0)
dec_mean, dec_median, dec_std = sigma_clipped_stats(dec, sigma=2.0)

# +
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(ra, dec, alpha=0.3, linewidths=0, label="matched sources")
ax.scatter(ra_median, dec_median, marker="x", s=400, label="sig-clip median")
ax.scatter(ra_mean, dec_mean, marker="x", s=400, label="sig-clip mean")
ax.errorbar(ra_mean, dec_mean, xerr=ra_std, yerr=dec_std, color="k", capsize=6, capthick=2, alpha=0.3, label="sig-clip stdev")
ax.axvline(0.0, color="k", lw=3, linestyle="dashed")
ax.axhline(0.0, color="k", lw=3, linestyle="dashed")
ax.legend(ncol=2)
ax.set(
    xlim=[-500, 200],
    ylim=[-500, 200],
    xlabel="displacement RA, milliarcsec", ylabel="displacement DEC, milliarcsec",
)

ax.set_aspect("equal")
# -

hdu2.header["CRVAL1"] -= Angle(ra_median * u.milliarcsecond).deg
hdu2.header["CRVAL2"] -= Angle(dec_median * u.milliarcsecond).deg
newname2 = f"align-{fname2}"
hdu2.writeto(datapath / f"{newname2}.fits", overwrite=True)

# ## Repeat for F658N

fname2 = "pcmos-f658n_sharp_16"
hdu2, = fits.open(datapath / f"{fname2}.fits")
w2 = WCS(hdu2.header)
stab2 = Table.read(datapath / f"{fname2}-sources-{sname2}.ecsv")
c2 = w2.pixel_to_world(stab2["xcentroid"], stab2["ycentroid"])

max_sep = 1.0 * u.arcsec
idx, d2d, d3d = c1.match_to_catalog_sky(c2)
sep_constraint = d2d < max_sep
c1_matches = c1[sep_constraint]
c2_matches = c2[idx[sep_constraint]]
f"Out of {len(c1)} sources in catalog #1 we have {len(c1_matches)} in catalog #2, but only {len(set(idx[sep_constraint]))} are unique."

stab1_matches = stab1[sep_constraint]
stab2_matches = stab2[idx[sep_constraint]]
ii1, ii2 = match_catalogs(c1, c2, max_sep=max_sep)
c1m = c1[ii1]
c2m = c2[ii2]
seps = c1m.separation(c2m)

offsets = c1m.spherical_offsets_to(c2m)
ra, dec = [_.milliarcsecond for _ in offsets]
max_sep_mas = max_sep.to(u.milliarcsecond).value
limits = [-max_sep_mas, max_sep_mas]

fig, ax = plt.subplots(figsize=(12, 4))
ax.hist([ra, dec], bins=31, label=["ra", "dec"])
ax.set(xlabel="separation, milliarcsec", yscale="linear", xlim=limits)
ax.axvline(0.0, color="k", lw=3, linestyle="dashed")
ax.legend()
...;

ra_mean, ra_median, ra_std = sigma_clipped_stats(ra, sigma=2.0)
dec_mean, dec_median, dec_std = sigma_clipped_stats(dec, sigma=2.0)

# +
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(ra, dec, alpha=0.3, linewidths=0, label="matched sources")
ax.scatter(ra_median, dec_median, marker="x", s=400, label="sig-clip median")
ax.scatter(ra_mean, dec_mean, marker="x", s=400, label="sig-clip mean")
ax.errorbar(ra_mean, dec_mean, xerr=ra_std, yerr=dec_std, color="k", capsize=6, capthick=2, alpha=0.3, label="sig-clip stdev")
ax.axvline(0.0, color="k", lw=3, linestyle="dashed")
ax.axhline(0.0, color="k", lw=3, linestyle="dashed")
ax.legend(ncol=2)
ax.set(
    xlim=[-500, 200],
    ylim=[-500, 200],
    xlabel="displacement RA, milliarcsec", ylabel="displacement DEC, milliarcsec",
)

ax.set_aspect("equal")
# -

hdu2.header["CRVAL1"] -= Angle(ra_median * u.milliarcsecond).deg
hdu2.header["CRVAL2"] -= Angle(dec_median * u.milliarcsecond).deg
newname2 = f"align-{fname2}"
hdu2.writeto(datapath / f"{newname2}.fits", overwrite=True)



# ## Repeat for F656N

fname2 = "pcmos-f656n_sharp_16"
hdu2, = fits.open(datapath / f"{fname2}.fits")
w2 = WCS(hdu2.header)
stab2 = Table.read(datapath / f"{fname2}-sources-{sname2}.ecsv")
c2 = w2.pixel_to_world(stab2["xcentroid"], stab2["ycentroid"])

max_sep = 1.0 * u.arcsec
idx, d2d, d3d = c1.match_to_catalog_sky(c2)
sep_constraint = d2d < max_sep
c1_matches = c1[sep_constraint]
c2_matches = c2[idx[sep_constraint]]
f"Out of {len(c1)} sources in catalog #1 we have {len(c1_matches)} in catalog #2, but only {len(set(idx[sep_constraint]))} are unique."

stab1_matches = stab1[sep_constraint]
stab2_matches = stab2[idx[sep_constraint]]
ii1, ii2 = match_catalogs(c1, c2, max_sep=max_sep)
c1m = c1[ii1]
c2m = c2[ii2]
seps = c1m.separation(c2m)

offsets = c1m.spherical_offsets_to(c2m)
ra, dec = [_.milliarcsecond for _ in offsets]
max_sep_mas = max_sep.to(u.milliarcsecond).value
limits = [-max_sep_mas, max_sep_mas]

fig, ax = plt.subplots(figsize=(12, 4))
ax.hist([ra, dec], bins=31, label=["ra", "dec"])
ax.set(xlabel="separation, milliarcsec", yscale="linear", xlim=limits)
ax.axvline(0.0, color="k", lw=3, linestyle="dashed")
ax.legend()
...;

ra_mean, ra_median, ra_std = sigma_clipped_stats(ra, sigma=1.5)
dec_mean, dec_median, dec_std = sigma_clipped_stats(dec, sigma=1.5)

# +
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(ra, dec, alpha=0.3, linewidths=0, label="matched sources")
ax.scatter(ra_median, dec_median, marker="x", s=400, label="sig-clip median")
ax.scatter(ra_mean, dec_mean, marker="x", s=400, label="sig-clip mean")
ax.errorbar(ra_mean, dec_mean, xerr=ra_std, yerr=dec_std, color="k", capsize=6, capthick=2, alpha=0.3, label="sig-clip stdev")
ax.axvline(0.0, color="k", lw=3, linestyle="dashed")
ax.axhline(0.0, color="k", lw=3, linestyle="dashed")
ax.legend(ncol=2)
ax.set(
    xlim=[-1000, 1000],
    ylim=[-1000, 1000],
    xlabel="displacement RA, milliarcsec", ylabel="displacement DEC, milliarcsec",
)

ax.set_aspect("equal")
# -

hdu2.header["CRVAL1"] -= Angle(ra_median * u.milliarcsecond).deg
hdu2.header["CRVAL2"] -= Angle(dec_median * u.milliarcsecond).deg
newname2 = f"align-{fname2}"
hdu2.writeto(datapath / f"{newname2}.fits", overwrite=True)


