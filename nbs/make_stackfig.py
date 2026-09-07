import gc
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')          # no display on a compute node
import matplotlib.pyplot as plt

from load_wus_d3 import load_domain_coords
from categorize import build_anomalies, make_ts, categorize

# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------
savedir = '/glade/work/mcowherd/'
figdir = '../figures'

inc = 30
y1, y2 = 1980, 1980 + inc          # baseline window 1980-2009
BASELINE_YEARS = range(y1, y2)
SNOW_THRESH_MM = 100               # 10 cm, per manuscript Section 2.1

bounds = {'SN': [-122, -118, 34.7, 41],
          'MR': [-114.3, -104, 42, 49]}

# (domain, region, drought_only) for each of the six panels
setup = [('d01', 'SN', False), ('d02', 'SN', False), ('d02', 'SN', True),
         ('d01', 'MR', False), ('d02', 'MR', False), ('d02', 'MR', True)]

color_list = ['gold', 'coral', 'darkred', 'lightpink',
              'darkgray', 'lightgray', 'mediumpurple']
label_list = ['dry drought', 'warm and dry drought', 'warm drought',
              'other drought', 'low normal', 'high normal', 'deluge']

drought_colors = ['gold', 'coral', 'darkred', 'lightpink']
drought_labels = ['dry drought', 'warm and dry drought', 'warm drought',
                  'other drought']

CHUNKS = {'gcm': 1, 'year': -1}    # elementwise across gcm until the final mean


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def load_mask_and_coords(domain, thresh_mm=SNOW_THRESH_MM):
    """Snow mask + 2D lat/lon for one domain."""
    path = os.path.join(savedir, f'allsnowmax_BC_{domain}.nc')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found -- run build_snow_mask.py for {domain} first.')

    ms = xr.open_dataset(path, chunks={'gcm': 1})
    mask = (ms.sel(year=slice(y1, y2 - 1)).mean(dim='year').mean(dim='gcm').swe
            > thresh_mm).compute()

    # build_snow_mask.py calls attach_coords, so XLAT/XLONG ride along on
    # the mask; drop them so they don't collide with la/lo below
    mask = mask.drop_vars(['XLAT', 'XLONG', 'elevation', 'landmask'],
                          errors='ignore')

    dc = load_domain_coords(domain=domain)
    lo = dc['XLONG'].rename({'south_north': 'lat', 'west_east': 'lon'})
    la = dc['XLAT'].rename({'south_north': 'lat', 'west_east': 'lon'})
    return mask, la, lo


def region_mask(region, la, lo, snowmask):
    lonmin, lonmax, latmin, latmax = bounds[region]
    return ((la > latmin) & (la < latmax) &
            (lo > lonmin) & (lo < lonmax) & snowmask)


# ---------------------------------------------------------------------
# one domain at a time: reduce to small series, then free
# ---------------------------------------------------------------------
series = {}          # panel index -> {category: (year,) DataArray}
diagnostics = []

for dom in sorted({s[0] for s in setup}):
    print(f'\n=== {dom} ===')

    cache = os.path.join(savedir, f'swei_peakmonth_all_gcm_{dom}_fixed.nc')
    if not os.path.exists(cache):
        raise FileNotFoundError(
            f'{cache} not found -- run build_swei.py for {dom} first.')

    swei = xr.open_dataset(cache, chunks=CHUNKS)['swei']
    anoms = build_anomalies(domain=dom, baseline_years=BASELINE_YEARS,
                            cache_dir=savedir).chunk(CHUNKS)
    snowmask, la, lo = load_mask_and_coords(dom)
    swei = swei.where(snowmask)

    print(f'{dom}: {int(snowmask.sum())} unmasked pixels, swei {swei.shape}')

    # sanity: baseline drought/wet fractions, computed lazily rather than
    # by pulling the whole array into memory
    base = swei.sel(year=slice(y1, y2 - 1))
    n_valid = int(base.notnull().sum().compute())
    n_dr = int((base <= -0.8).sum().compute())
    n_wet = int((base >= 0.8).sum().compute())
    print(f'{dom}: baseline drought frac {n_dr / n_valid:.3f} (expect ~0.20), '
          f'wet frac {n_wet / n_valid:.3f}')

    for i, (d, region, drought_only) in enumerate(setup):
        if d != dom:
            continue
        rmask = region_mask(region, la, lo, snowmask)
        cats = make_ts(swei, anoms, rmask, drought_only=drought_only)
        series[i] = {k: v.compute() for k, v in cats.items()}
        print(f'  panel {chr(i + 97)}: {region} {dom} '
              f'{"(drought only)" if drought_only else ""}')

        # droughts that are neither warm nor dry have no bin in the
        # manuscript typology -- quantify rather than silently drop
        if drought_only:
            c = categorize(swei, anoms)
            other = float(c['drought_other'].where(rmask).mean().compute())
            tot = float((c['dry'] + c['warmdry'] + c['warm'] +
                         c['drought_other']).where(rmask).mean().compute())
            diagnostics.append(
                f'{region} {dom}: uncategorized droughts {other:.4f} of area, '
                f'{100 * other / tot:.1f}% of all droughts')

       


    del swei, anoms, snowmask, la, lo
    gc.collect()

# ---------------------------------------------------------------------
# plot (only the small series are alive now)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(25, 10))
axs = ax.flatten()

for i, (dom, region, drought_only) in enumerate(setup):
    cats = series[i]
    labels = drought_labels if drought_only else label_list
    colors = drought_colors if drought_only else color_list

    yrs = cats[list(cats)[0]].year.values
    axs[i].stackplot(yrs, list(cats.values()), labels=labels, colors=colors)
    axs[i].set_ylim((0, 1))
    axs[i].set_xlim((1981, 2099))
    if i in (1, 4):
        axs[i].set_yticklabels([])
    if i < 3:
        axs[i].set_xticklabels([])

axs[0].set_ylabel('proportion of area', fontsize=20)
axs[3].set_ylabel('proportion of area', fontsize=20)
axs[2].set_ylabel('proportion of drought', fontsize=20)
axs[5].set_ylabel('proportion of drought', fontsize=20)
axs[0].text(1953, 0.150, 'Sierra Nevada', fontsize=28, rotation=90)
axs[3].text(1953, 0.150, 'Middle Rockies', fontsize=28, rotation=90)
for i in range(3, 6):
    axs[i].set_xlabel('year')
axs[4].legend(loc='lower center', bbox_to_anchor=(0.4, -0.45), ncol=7, fontsize=20)
axs[0].set_title('45 km', fontsize=28)
axs[1].set_title('9 km', fontsize=28)
axs[2].set_title('9 km', fontsize=28)
for i in range(6):
    axs[i].text(1983, 1.03, chr(i + 97))

os.makedirs(figdir, exist_ok=True)
outpath = os.path.join(figdir, 'fig_stack_fixed.jpg')
plt.savefig(outpath, dpi=400, bbox_inches='tight')
print(f'\nwrote {outpath}')

for region, (i45, i9) in {'SN': (0, 1), 'MR': (3, 4)}.items():
    d45 = sum(series[i45][k] for k in ('dry', 'warmdry', 'warm', 'drought_other'))
    d9  = sum(series[i9][k]  for k in ('dry', 'warmdry', 'warm', 'drought_other'))
    for name, yrs in [('baseline', slice(y1, y2 - 1)), ('eoc', slice(2070, 2099))]:
        diff = (d45 - d9).sel(year=yrs)
        print(f'{region} {name}: 45-9 km {float(diff.mean()):+.3f} +/- {float(diff.std()):.3f}')

# ---------------------------------------------------------------------
# numbers for the text
# ---------------------------------------------------------------------
print('\n--- diagnostics ---')
for line in diagnostics:
    print(line)

print('\n--- drought area, baseline vs end-of-century ---')
for i, (dom, region, drought_only) in enumerate(setup):
    if drought_only:
        continue
    cats = series[i]
    total_drought = (cats['dry'] + cats['warmdry'] + cats['warm'] +
                 cats['drought_other'])
    base = total_drought.sel(year=slice(y1, y2 - 1))
    eoc = total_drought.sel(year=slice(2070, 2099))
    print(f'{region} {dom}: baseline {float(base.mean()):.3f} '
          f'+/- {float(base.std()):.3f}, '
          f'end-of-century {float(eoc.mean()):.3f} +/- {float(eoc.std()):.3f}')



print('\n--- Figure 2 statistics, 9 km ---')
PANELS = {'Sierra Nevada': 1, 'Middle Rockies': 4}   # d02, area fractions

for region, i in PANELS.items():
    c = series[i]
    drought = c['dry'] + c['warmdry'] + c['warm'] + c['drought_other']

    for name, yrs in [('baseline', slice(y1, y2 - 1)),
                      ('2070-2099', slice(2070, 2099))]:
        d = drought.sel(year=yrs)
        print(f'{region} {name}: drought area {float(d.mean()):.3f} '
              f'+/- {float(d.std()):.3f} (interannual), '
              f'range {float(d.min()):.3f}-{float(d.max()):.3f}, '
              f'wet {float(c["wet"].sel(year=yrs).mean()):.3f}')

        # shares of drought, both aggregations
        for k in ('dry', 'warmdry', 'warm', 'drought_other'):
            pooled = float(c[k].sel(year=yrs).mean() / d.mean())
            annual = float((c[k].sel(year=yrs) / d).mean())
            print(f'    {k:14s} pooled {pooled:.3f}   mean-of-annual {annual:.3f}')

        trel_p = float((c['warm'] + c['warmdry']).sel(year=yrs).mean() / d.mean())
        warmonly_p = float(c['warm'].sel(year=yrs).mean() /
                           (c['warm'] + c['warmdry']).sel(year=yrs).mean())
        print(f'    temperature-related {trel_p:.3f}; '
              f'of those, warm-only {warmonly_p:.3f}')

        trel = (c['warm'] + c['warmdry']).sel(year=yrs) / d
        print(f'    temperature-related mean-of-annual {float(trel.mean()):.3f} '
                      f'+/- {float(trel.std()):.3f}')


# ---------------------------------------------------------------------
# SWEI floor: with a 30-year baseline and the Gringorten plotting
# position, rank 1 maps to p = 0.56/30.12 and SWEI = -2.084, so no year
# can score below that however anomalous it is. Report how much of the
# end-of-century distribution sits at that floor, since severity there
# is censored rather than resolved.
# ---------------------------------------------------------------------
from scipy.stats import norm

FLOOR = norm.ppf((1 - 0.44) / (30 + 0.12))          # -2.0839
print(f'\n--- SWEI floor ({FLOOR:.3f}), 9 km ---')

swei = xr.open_dataset(
    os.path.join(savedir, 'swei_peakmonth_all_gcm_d02_fixed.nc'),
    chunks=CHUNKS)['swei']
snowmask, la, lo = load_mask_and_coords('d02')

for region, name in [('SN', 'Sierra Nevada'), ('MR', 'Middle Rockies')]:
    rmask = region_mask(region, la, lo, snowmask)
    e = swei.where(rmask).sel(year=slice(2070, 2099))

    n = int(e.notnull().sum().compute())
    f = int((e <= FLOOR + 1e-6).sum().compute())

    v = e.values.ravel()
    v = v[np.isfinite(v)]
    print(f'{name}: {100 * f / n:.1f}% of end-of-century grid-cell-years '
          f'at the floor (n={n})')
    print(f'    mean {v.mean():.2f} +/- {v.std():.2f}, '
          f'median {np.median(v):.2f}, '
          f'IQR {np.percentile(v, 25):.2f} to {np.percentile(v, 75):.2f}')

    # severity: mean SWEI over drought events only, as line 348 defines it
    d = e.where(e <= -0.8).values.ravel()
    d = d[np.isfinite(d)]
    print(f'    drought-only: mean {d.mean():.2f} +/- {d.std():.2f}, '
          f'median {np.median(d):.2f}, '
          f'{100 * (d <= FLOOR + 1e-6).mean():.1f}% at floor')

del swei, snowmask, la, lo
gc.collect()