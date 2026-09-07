#!/usr/bin/env python
"""
Compute change in November-April total snowfall per degree of global warming
(dSF/dT) for the WUS-D3 ensemble, from WRF's own `prec_snow` output.

    dSF/dT (x, m) = [ mean Nov-Apr snowfall(x, m, future window)
                      - mean Nov-Apr snowfall(x, m, reference window) ]
                    / [ global mean 2 m T(m, future window)
                        - global mean 2 m T(m, reference window) ]

Each ESM is normalized by its own global warming; the ensemble mean is taken
afterwards, so inter-model spread in climate sensitivity is removed rather
than averaged over.

`prec_snow` is the model's own phase-partitioned solid precipitation, so no
rain-snow threshold is imposed here. Note that `snow` in the same archive is
snow water equivalent (a state), not snowfall (a flux).

Snowfall is summed over 1 November - 30 April and labelled by the water year
containing April, matching the manuscript's winter definition. Only the years
needed for the three windows are read, not the full record.

The snow MASK still comes from baseline peak SWE > 10 cm (Section 2.1), read
from allsnowmax_BC_{domain}.nc.

Writes dsfdt_{domain}.nc:
    dsfdt         (gcm, lat2d, lon2d)  mm K-1, end-of-century
    dsfdt_midc    (gcm, lat2d, lon2d)  mm K-1, mid-century
    sf_ref        (gcm, lat2d, lon2d)  mm, reference-window Nov-Apr snowfall
    baseline_sf   (lat2d, lon2d)       mm, ensemble mean of sf_ref
    baseline_swe  (lat2d, lon2d)       mm, ensemble-mean reference peak SWE
    snowmask      (lat2d, lon2d)       bool
    dT, dT_midc   (gcm,)               K

Usage:
    python compute_dsfdt_wrf.py --domain d02 --outdir /glade/work/mcowherd/dsfdt/
    python compute_dsfdt_wrf.py --domain d02 --dry-run     # check paths only
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr


ROOT = '/glade/campaign/uwyo/wyom0169/wus-d3/postprocess'

# short name -> variant label used in directory and file names
VARIANTS = {'cesm2': 'r11i1p1f1',
            'mpi-esm1-2-lr': 'r7i1p1f1',
            'cnrm-esm2-1': 'r1i1p1f2',
            'ec-earth3-veg': 'r1i1p1f1',
            'fgoals-g3': 'r1i1p1f1',
            'canesm5': 'r1i1p2f1',
            'access-cm2': 'r5i1p1f1',
            'ec-earth3': 'r1i1p1f1'}

GCMS = list(VARIANTS)          # the eight ESMs in Table 1; UKESM excluded

ECS = {'access-cm2': 4.66, 'canesm5': 5.64, 'cesm2': 5.15,
       'cnrm-esm2-1': 4.79, 'ec-earth3': 4.26, 'ec-earth3-veg': 4.33,
       'fgoals-g3': 2.87, 'mpi-esm1-2-lr': 3.03}

VAR = 'prec_snow'
SNOW_THRESH_MM = 100.0
REF_WINDOW  = (1981, 2010)   
EOC_WINDOW  = (2070, 2099)
MIDC_WINDOW = (2030, 2059)

DEFAULT_MAXSNOW_DIR = '/glade/campaign/uwyo/wyom0200/berkeley/'
DEFAULT_TASFILE = 'wrfrun_globalavg_temp.csv'


# --------------------------------------------------------------------------
# file discovery
# --------------------------------------------------------------------------

def year_files(gcm, domain, years):
    """Paths to prec_snow files covering `years`, across hist and ssp dirs.

    Globs rather than constructing names, so a change in the experiment or
    bias-correction label doesn't silently return nothing.
    """
    variant = VARIANTS[gcm]
    out = []
    for exp in ('historical', 'ssp370'):
        d = os.path.join(ROOT, f'{gcm}_{variant}_{exp}_bc', 'postprocess', domain)
        if not os.path.isdir(d):
            continue
        for y in years:
            out.extend(glob.glob(os.path.join(d, f'{VAR}.daily.*.{domain}.{y}.nc')))
    return sorted(set(out))


def needed_years(windows):
    """Calendar years to read: each water year needs the prior Nov-Dec."""
    ys = set()
    for lo, hi in windows:
        ys.update(range(lo - 1, hi + 1))
    return sorted(ys)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _month_year(times):
    m = np.array([int(t.month) if hasattr(t, 'month')
                  else int(pd.Timestamp(t).month) for t in times])
    y = np.array([int(t.year) if hasattr(t, 'year')
                  else int(pd.Timestamp(t).year) for t in times])
    return m, y


def seasonal_snowfall(gcm, domain, years, chunks, min_days=150):
    """Nov-Apr total snowfall by water year for one ESM, (year, lat2d, lon2d).

    Water years with fewer than `min_days` days present are dropped; a full
    Nov-Apr window is 181 days (176 on a 360-day calendar).
    """
    files = year_files(gcm, domain, years)
    if not files:
        raise FileNotFoundError(f'{gcm} {domain}: no {VAR} files for {years[0]}-{years[-1]}')

    ds = xr.open_mfdataset(files, combine='by_coords', chunks=chunks,
                           parallel=False)
    if VAR in ds:
        da = ds[VAR]
    else:
        cands = [v for v in ds.data_vars if ds[v].ndim >= 3]
        if len(cands) != 1:
            raise KeyError(f'{gcm}: cannot identify snowfall among {list(ds.data_vars)}')
        da = ds[cands[0]]

    tname = 'time' if 'time' in da.dims else 'day'
    if tname != 'time':
        da = da.rename({tname: 'time'})

    months, cal_years = _month_year(da.time.values)
    wy = np.where(months >= 10, cal_years + 1, cal_years)   # water year
    keep = np.where((months >= 11) | (months <= 4))[0]

    da = da.isel(time=keep)
    da = da.assign_coords(year=('time', wy[keep]))
    total = da.groupby('year').sum('time')

    # Drop water years missing part of Nov-Apr. Without this, a year whose
    # preceding Nov-Dec files are absent (1979 is not in the archive, so
    # water year 1980 is Jan-Apr only) silently returns a low seasonal total.
    ndays = xr.DataArray(np.ones(keep.size), dims='time',
                         coords={'year': ('time', wy[keep])}).groupby('year').sum('time')
    complete = ndays >= min_days
    dropped = [int(y) for y in ndays.year.values[~complete.values]]
    if dropped:
        print(f'    incomplete water years dropped: {dropped}')
    return total.sel(year=ndays.year.values[complete.values])


def load_peak_swe(domain, maxsnow_dir):
    path = os.path.join(maxsnow_dir, f'allsnowmax_BC_{domain}.nc')
    if not os.path.exists(path):
        sys.exit(f'not found: {path}')
    ds = xr.open_dataset(path)
    name = next((c for c in ('__xarray_dataarray_variable__', 'swe', 'snow')
                 if c in ds), None)
    if name is None:
        sys.exit(f'{path}: cannot identify the SWE variable among {list(ds)}')
    da = ds[name]
    if 'time' in da.dims:
        _, yrs = _month_year(da.time.values)
        da = da.assign_coords(time=yrs).rename({'time': 'year'})

    order = ['cesm2', 'mpi-esm1-2-lr', 'cnrm-esm2-1', 'ec-earth3-veg',
             'fgoals-g3', 'ukesm1-0-ll', 'canesm5', 'access-cm2', 'ec-earth3']
    n = da.sizes['gcm']
    if n == len(order):
        da = da.assign_coords(gcm=order)
    elif n == len(order) - 1:
        da = da.assign_coords(gcm=[g for g in order if g != 'ukesm1-0-ll'])
    else:
        sys.exit(f'{path}: unexpected gcm axis length {n}')
    for c in ('XLAT', 'XLONG', 'elevation', 'landmask'):
        if c in da.coords:
            da = da.drop_vars(c)
    return da.sel(gcm=[g for g in da.gcm.values if g in GCMS]).rename('swe')


def load_global_tas(tasfile):
    if not os.path.exists(tasfile):
        sys.exit(f'not found: {tasfile}')
    df = pd.read_csv(tasfile, index_col=0, parse_dates=True)
    for col in ('source_id', 'year', 'tas'):
        if col not in df.columns:
            sys.exit(f'{tasfile}: expected {col!r}, found {list(df.columns)}')
    known = {g.lower().replace('_', '-'): g for g in GCMS}
    df = df.copy()
    df['gcm'] = df.source_id.map(
        lambda r: known.get(str(r).strip().lower().replace('_', '-')))
    unknown = sorted(set(df.source_id[df.gcm.isna()]))
    if unknown:
        print(f'  note: ignoring source_id {unknown}')
    df = df[df.gcm.notna()]
    return df.groupby(['gcm', 'year'])['tas'].mean().to_xarray()


# --------------------------------------------------------------------------

def window_mean(da, window):
    lo, hi = window
    sub = da.sel(year=slice(lo, hi))
    if sub.sizes['year'] == 0:
        sys.exit(f'no years in {lo}-{hi}; data spans '
                 f'{int(da.year.min())}-{int(da.year.max())}')
    return sub.mean('year')


def compute(args):
    ref, eoc, midc = tuple(args.ref), tuple(args.future), tuple(args.midc)
    years = needed_years([ref, eoc, midc])
    chunks = {'time': args.time_chunk}

    print(f'domain {args.domain}, reading {len(years)} years per ESM')
    tas = load_global_tas(args.tasfile)
    swe = load_peak_swe(args.domain, args.maxsnow_dir)

    refs, eocs, midcs, used = [], [], [], []
    for gcm in GCMS:
        if gcm not in set(tas.gcm.values):
            print(f'  {gcm}: no global temperature, skipped')
            continue
        try:
            sf = seasonal_snowfall(gcm, args.domain, years, chunks)
        except (FileNotFoundError, KeyError) as e:
            print(f'  {e}')
            continue
        refs.append(window_mean(sf, ref).compute())
        eocs.append(window_mean(sf, eoc).compute())
        midcs.append(window_mean(sf, midc).compute())
        used.append(gcm)
        print(f'  {gcm}: done')

    if len(used) < 2:
        sys.exit('fewer than two ESMs loaded')

    dim = xr.DataArray(used, dims='gcm', name='gcm')
    sf_ref = xr.concat(refs, dim=dim)
    sf_eoc = xr.concat(eocs, dim=dim)
    sf_midc = xr.concat(midcs, dim=dim)

    tas = tas.sel(gcm=used)
    dT = window_mean(tas, eoc) - window_mean(tas, ref)
    dT_midc = window_mean(tas, midc) - window_mean(tas, ref)
    if float(dT.min()) < 0.5:
        sys.exit(f'end-of-century warming below 0.5 K for some ESM '
                 f'({float(dT.min()):.2f} K); ratio would be unstable')

    baseline_swe = window_mean(swe.sel(gcm=used), ref).mean('gcm')

    out = xr.Dataset({'dsfdt': ((sf_eoc - sf_ref) / dT).astype('float32'),
                      'dsfdt_midc': ((sf_midc - sf_ref) / dT_midc).astype('float32'),
                      'sf_ref': sf_ref.astype('float32'),
                      'baseline_sf': sf_ref.mean('gcm').astype('float32'),
                      'baseline_swe': baseline_swe.astype('float32'),
                      'snowmask': baseline_swe > args.thresh,
                      'dT': dT.astype('float32'),
                      'dT_midc': dT_midc.astype('float32')})
    out.dsfdt.attrs = {'units': 'mm K-1', 'long_name':
                       'change in Nov-Apr snowfall per degree global warming'}
    out.baseline_sf.attrs = {'units': 'mm'}
    out.attrs = {'quantity': f'November-April total {VAR} (WRF phase partitioning)',
                 'reference_window': f'{ref[0]}-{ref[1]}',
                 'eoc_window': f'{eoc[0]}-{eoc[1]}',
                 'midc_window': f'{midc[0]}-{midc[1]}',
                 'snow_mask': f'baseline peak SWE > {args.thresh} mm',
                 'normalization': 'per-ESM global dT applied before ensemble mean',
                 'created_by': os.path.basename(__file__)}
    return out


def report(out):
    m = out.snowmask
    base = out.baseline_sf.where(m)
    print('\n  per-ESM warming and implied snowfall loss')
    print(f'  {"ESM":16s} {"ECS":>5s} {"dT":>6s} {"med dSF/dT":>11s} {"loss/baseline":>14s}')
    for g in out.gcm.values:
        v = out.dsfdt.sel(gcm=g).where(m)
        dT = float(out.dT.sel(gcm=g))
        print(f'  {str(g):16s} {ECS.get(str(g), float("nan")):5.2f} {dT:6.2f} '
              f'{float(v.median()):11.1f} '
              f'{float((-(v * dT) / base).median()):14.2f}')

    ens = out.dsfdt.mean('gcm').where(m)
    dT_mean = float(out.dT.mean())
    pct = float((100 * ens / base).median())
    frac = float((-(ens * dT_mean) / base).median())
    print('\n  ensemble mean, snow-masked cells')
    print(f'    median dSF/dT    {float(ens.median()):8.1f} mm/K')
    print(f'    median %/K       {pct:8.1f} %/K')
    print(f'    implied loss     {100 * frac:8.1f} % of baseline snowfall')
    print(f'    mean dT          {dT_mean:8.2f} K')
    print(f'    ceiling on |%/K| {100 / dT_mean:8.1f}')
    print(f'    masked cells     {int(m.sum()):8d} of {m.size}')
    print('\n  ' + ('FAIL: loss exceeds the entire baseline snowfall.'
                    if abs(pct) > 100 / dT_mean else
                    'OK: loss rates are physically attainable.'))
    print('  Compare the implied loss against the regional snowfall declines '
          'already reported in Figure 6 (Sierra Nevada 49.6%, '
          'Middle Rockies 25.8% at 9 km).')


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--domain', default='d02')
    p.add_argument('--outdir', default='.')
    p.add_argument('--tasfile', default=DEFAULT_TASFILE)
    p.add_argument('--maxsnow-dir', default=DEFAULT_MAXSNOW_DIR)
    p.add_argument('--ref', nargs=2, type=int, default=REF_WINDOW)
    p.add_argument('--future', nargs=2, type=int, default=EOC_WINDOW)
    p.add_argument('--midc', nargs=2, type=int, default=MIDC_WINDOW)
    p.add_argument('--thresh', type=float, default=SNOW_THRESH_MM)
    p.add_argument('--time-chunk', type=int, default=90)
    p.add_argument('--dry-run', action='store_true',
                   help='report how many files each ESM would read, then exit')
    args = p.parse_args()

    if args.dry_run:
        years = needed_years([tuple(args.ref), tuple(args.future), tuple(args.midc)])
        print(f'{args.domain}: {len(years)} calendar years '
              f'({years[0]}-{years[-1]}, non-contiguous)')
        for gcm in GCMS:
            f = year_files(gcm, args.domain, years)
            print(f'  {gcm:16s} {len(f):4d} files' +
                  (f'   e.g. {os.path.basename(f[0])}' if f else '   NONE FOUND'))
        return

    out = compute(args)
    report(out)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f'dsfdt_{args.domain}.nc')
    out.to_netcdf(path)
    print(f'\n  wrote {path}')


if __name__ == '__main__':
    main()