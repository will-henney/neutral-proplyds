import sys
import os
import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
from matplotlib import pyplot as plt
import seaborn as sns

try: 
    infile = sys.argv[1]
except:
    sys.exit('Usage: {} FITSFILE'.format(sys.argv[0]))


basename = os.path.basename(infile)
baseroot, _ = os.path.splitext(basename)
figfile = f"{sys.argv[0]}-{baseroot}.pdf"

hdu = fits.open(infile)[0]
if hdu.data is None:
    hdu = fits.open(infile)[1]
hdr = hdu.header

ny, nx = hdu.data.shape

# Size of chunks
mx, my = 290, 290
xchunks, ychunks = nx//mx, ny//my


fig, axes = plt.subplots(ychunks, xchunks,
                         sharex=True, sharey=True,
                         figsize=(10, 10),
)


hdu.data /= 1000.0

m = np.isfinite(hdu.data)
vmin, vmax = np.percentile(hdu.data[m], [1.0, 99.0])
vrange = vmax - vmin
vmin -= 0.3*vrange
vmax += 0.3*vrange

m = m & (hdu.data >= vmin) & (hdu.data <= vmax)

fitter = fitting.LevMarLSQFitter()
for jchunk in range(ychunks):
    yslice = slice(jchunk*my, jchunk*my + my)
    for ichunk in range(xchunks):
        xslice = slice(ichunk*mx, ichunk*mx + mx)

        mm = m[yslice, xslice]
        tile = hdu.data[yslice, xslice][mm]
        ax = axes[ychunks - jchunk - 1, ichunk]
        hist, edges, _ = ax.hist(tile, bins=100, range=[vmin, vmax])
        centers = 0.5*(edges[:-1] + edges[1:])


        a0 = hist.max()
        v0 = np.mean(tile)
        vmedian = np.median(tile)
        s0 = np.std(tile)
        g_init = models.Gaussian1D(amplitude=a0, mean=v0, stddev=s0)
        select = hist > 0.3*a0
        g = fitter(g_init, centers[select], hist[select])
        ax.plot(centers, g(centers), c='r', lw=0.5)
        # ax.plot(centers, g_init(centers), c='g')

        ax.axvline(0.0, c='k', alpha=0.5)
        ax.axvline(vmedian, c='r', alpha=1.0)

        s = f"peak = {g.mean.value:.2f}\nstd = {g.stddev.value:.2f}"
        ax.text(0.95, 0.95, s,
                ha='right', va='top',
                fontsize='xx-small',
                transform=ax.transAxes)

fig.savefig(figfile)
print(figfile, end='')
