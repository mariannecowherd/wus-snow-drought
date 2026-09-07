"""
Drought categorization (warm / dry / warm-and-dry) and area-fraction
time series for Figure 2.

Replaces myutil.py's make_ts, which read the old categs_snow_*_BC.nc
files produced by the buggy whole-record SWEI ranking. This version
works from the corrected per-GCM SWEI cache plus winter temperature and
precipitation anomalies computed here.

Categories follow Section 2.2: a snow drought (SWEI <= -0.8) is
classified by the sign of its coincident 1 Nov - 30 Apr temperature and
precipitation anomalies relative to the 1980-2010 baseline mean at that
pixel, for that GCM.

NOTE the old fig_stack.ipynb added a flat +0.02 to the wet category
before plotting, apparently to keep it visible in the stack. That is a
two-percentage-point inflation baked into the figure and is NOT
reproduced here -- do not add it back.
"""

import glob
import os

import numpy as np
import xarray as xr

from load_wus_d3 import (GCM_VARIANTS, CALENDARS, _experiment_dir,
                          FILENAME_EXPERIMENT_ALIASES, _fix_time_coord,
                          _standardize_dims, attach_coords)

WINTER_MONTHS = [11, 12, 1, 2, 3, 4]


def _find_var_year_files(var, gcm, variant, experiment, domain, bc=True):
    """
    Glob directly for {var}.daily.* files.

    An earlier version derived these paths by string-substituting the
    variable name into the file list found for 'snow'. That silently
    produced nonexistent or mismatched paths whenever a variable's files
    were named or located differently, with no error raised until much
    later -- globbing and failing loudly is the point of this function.
    """
    expdir = os.path.join(_experiment_dir(gcm, variant, experiment, bc=bc),
                          'postprocess', domain)
    fexp = FILENAME_EXPERIMENT_ALIASES.get(experiment, experiment)
    pattern = os.path.join(
        expdir, f'{var}.daily.{gcm}.{variant}.{fexp}.bias-correct.{domain}.*.nc')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'no files matched {pattern}')
    return files


def load_winter_var(var, gcm, domain='d02', variant=None, bc=True,
                     how='mean', pdd_base_c=0.0, var_name=None):
    """
    Load one variable's daily WUS-D3 output and reduce it to a single
    1 Nov - 30 Apr value per water year.

    how : 'mean' | 'sum' | 'pdd'
        'mean' -- winter mean (temperature)
        'sum'  -- winter total (precipitation)
        'pdd'  -- positive degree days: sum of daily degrees above
                  pdd_base_c, matching the PDD definition used in
                  Section 2.4. Use this if the categorization should be
                  consistent with the regression analysis rather than
                  with a plain winter-mean temperature.

    var_name : name of the variable INSIDE the file, if it differs from
        the filename prefix `var`. Not assumed to match -- this was true
        for 'snow' but is not guaranteed for t2/prec.
    """
    variant = variant or GCM_VARIANTS[gcm]
    calendar = CALENDARS[gcm]
    var_name = var_name or var

    def _pre(ds):
        return _fix_time_coord(ds, calendar=calendar)

    frames = []
    for experiment in ('historical', 'ssp370'):
        files = _find_var_year_files(var, gcm, variant, experiment, domain, bc=bc)
        frames.append(xr.open_mfdataset(files, combine='by_coords',
                                         decode_times=False, preprocess=_pre))
    ds = xr.concat(frames, dim='time').sortby('time')
    ds = _standardize_dims(ds)

    if var_name not in ds.data_vars:
        raise KeyError(
            f"'{var_name}' not found in {var}.daily files; available: "
            f"{list(ds.data_vars)} -- pass var_name= explicitly")

    da = ds[var_name]

    # water year = Oct(Y-1) .. Sep(Y), labelled by its ending calendar year
    wy = xr.where(da['time'].dt.month >= 10,
                  da['time'].dt.year + 1, da['time'].dt.year)
    da = da.assign_coords(water_year=('time', wy.data))
    da = da.sel(time=da['time'].dt.month.isin(WINTER_MONTHS))

    if how == 'pdd':
        # crude Kelvin check so the 0 C threshold means what it says
        if float(da.isel(time=0).max()) > 200:
            da = da - 273.15
        da = (da - pdd_base_c).clip(min=0)
        out = da.groupby('water_year').sum(dim='time')
    elif how == 'sum':
        out = da.groupby('water_year').sum(dim='time')
    elif how == 'mean':
        out = da.groupby('water_year').mean(dim='time')
    else:
        raise ValueError(f"how must be 'mean', 'sum' or 'pdd', got {how!r}")

    return out.rename({'water_year': 'year'})


def build_anomalies(domain='d02', baseline_years=range(1980, 2010),
                     cache_dir=None, gcm_variants=None,
                     temp_how='mean', t_var='t2', p_var='prec',
                     t_var_name=None, p_var_name=None):
    """
    Winter T and P anomalies from the baseline mean, per GCM per pixel
    per water year. Cached, since this is as expensive as the SWEI
    computation itself.

    temp_how : 'mean' (winter mean temperature, as Section 2.2 reads) or
        'pdd' (positive degree days, consistent with Section 2.4). These
        give different categorizations -- pick one and state it in the
        text rather than leaving the two sections inconsistent.

    Returns a Dataset with 't_anom' and 'p_anom' on (gcm, year, lat, lon).
    """
    gcm_variants = gcm_variants or GCM_VARIANTS
    baseline_years = list(baseline_years)
    cache_path = (os.path.join(cache_dir, f'winter_anoms_{domain}.nc')
                  if cache_dir else None)
    if cache_path and os.path.exists(cache_path):
        print(f'loading cached anomalies from {cache_path}')
        return xr.open_dataset(cache_path)

    t_list, p_list, names = [], [], []
    for gcm, variant in gcm_variants.items():
        print(f'{gcm}: winter T and P ...')
        t = load_winter_var(t_var, gcm, domain=domain, variant=variant,
                            how=temp_how, var_name=t_var_name)
        p = load_winter_var(p_var, gcm, domain=domain, variant=variant,
                            how='sum', var_name=p_var_name)

        missing = set(baseline_years) - set(t['year'].values.tolist())
        if missing:
            raise ValueError(f'{gcm}: baseline years absent from t2: '
                             f'{sorted(missing)}')

        t_list.append((t - t.sel(year=baseline_years).mean(dim='year')).compute())
        p_list.append((p - p.sel(year=baseline_years).mean(dim='year')).compute())
        names.append(gcm)
        del t, p

    out = xr.Dataset({
        't_anom': xr.concat(t_list, dim='gcm').assign_coords(gcm=names),
        'p_anom': xr.concat(p_list, dim='gcm').assign_coords(gcm=names),
    })
    out = attach_coords(out, domain=domain)

    # sanity: baseline anomalies are zero-mean by construction, and
    # end-of-century warming should be clearly positive nearly everywhere.
    # If these don't hold, the anomalies are wrong and so is Figure 2.
    b = float(out['t_anom'].sel(year=baseline_years).mean())
    print(f'  check: baseline t_anom mean {b:.4f} (expect ~0)')
    yrs = out['year'].values
    if yrs.max() >= 2099:
        e = float(out['t_anom'].sel(year=slice(2070, 2099)).mean())
        f = float((out['t_anom'].sel(year=slice(2070, 2099)) > 0).mean())
        print(f'  check: 2070-2099 t_anom mean {e:.3f}, '
              f'fraction warm {f:.3f} (expect clearly positive, near 1)')

    if cache_path:
        print(f'caching to {cache_path}')
        out.to_netcdf(cache_path)
    return out


def categorize(swei, anoms):
    """
    Boolean masks for each SWEI/anomaly category, on
    (gcm, year, lat, lon). Thresholds per Section 2.2.

    Droughts that are neither warm nor dry (temperature anomaly <= 0 AND
    precipitation anomaly >= 0) fall into none of the three drought
    categories -- the manuscript's typology has no bin for them. They are
    returned as 'drought_other' so they can be counted rather than
    silently dropped; check whether this is non-negligible before
    reporting drought-type fractions.
    """
    valid = swei.notnull()
    drought = (swei <= -0.8) & valid
    warm = anoms['t_anom'] > 0
    dry = anoms['p_anom'] < 0

    return {
        'dry':      (drought & dry & ~warm).where(valid),
        'warmdry':  (drought & dry & warm).where(valid),
        'warm':     (drought & warm & ~dry).where(valid),
        'lownorm':  ((swei > -0.8) & (swei <= 0)).where(valid),
        'highnorm': ((swei > 0) & (swei < 0.8)).where(valid),
        'wet':      (swei >= 0.8).where(valid),
        'drought_other': (drought & ~warm & ~dry).where(valid),
    }


def make_ts(swei, anoms, region_mask, drought_only=False):
    """
    Area fraction in each category per year, averaged across GCMs --
    i.e. thresholded per GCM first, then averaged, which is the correct
    order and matches what the old make_ts already did.

    drought_only=True renormalizes the three drought categories to sum
    to 1, for the third column of Figure 2. Note this DROPS
    'drought_other'; if that category is non-negligible, panels c and f
    do not represent the full drought area.
    """
    cats = categorize(swei, anoms)
    keys = (['dry', 'warmdry', 'warm', 'drought_other'] if drought_only
            else ['dry', 'warmdry', 'warm', 'drought_other',
                  'lownorm', 'highnorm', 'wet'])

    out = {k: cats[k].where(region_mask).mean(dim=['gcm', 'lat', 'lon'])
           for k in keys}

    if drought_only:
        total = sum(out.values())
        out = {k: v / total for k, v in out.items()}
    return out