"""!
Common utility functions and shared constants/instantiations.
"""

import datetime
import glob
import os

import dask
import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from rich.console import Console

from dirs import basedir, wrfdir, coorddir, workdir

##! Shared logging console object # noqa: E265
console = Console()

# CONSTANTS
MM_TO_IN = 0.03937008

# Keep global variables available to all modules
## define directories
# TODO remove these and use dirs.py instead

def read_merge(dir,domain,var,gcm,variant,date_start_hist, \
                date_start_ssp,date_end_hist,date_end_ssp,exp,bc,calendar):
    
    dir_x = "%s/%s_%s_historical/postprocess/" %(dir,gcm,variant) + domain + "/"
    
    if bc == True:
        dir_x = "%s/%s_%s_historical_bc/postprocess/" %(dir,gcm,variant) + domain + "/"

    if calendar == '360_day':
        date_end_hist = date_end_hist.split("-")
        date_end_hist = "%s-%s-%s" %(date_end_hist[0],
                                     date_end_hist[1],
                                     int( int(date_end_hist[2]) + 0.0001 - 1) )
        
        date_end_ssp = date_end_ssp.split("-")
        date_end_ssp = "%s-%s-%s" %(date_end_ssp[0],
                                     date_end_ssp[1],
                                     int( int(date_end_ssp[2]) + 0.0001 - 1) )
        
    print (calendar,date_end_hist,date_end_ssp)
    var_wrf = wrfread_gcm("hist",gcm,variant,dir_x,var,domain,calendar)
    var_hist = var_wrf.sel(day=slice(date_start_hist,date_end_hist))
    dir_x = "%s/%s_%s_%s/postprocess/" %(dir,gcm,variant,exp) + domain + "/"
    
    print (gcm,variant,bc,exp)
    
    if bc == True:
        dir_x = "%s/%s_%s_%s_bc/postprocess/" %(dir,gcm,variant,exp) + domain + "/"

    var_wrf = wrfread_gcm(exp,gcm,variant,dir_x,var,domain,calendar)
    var_ssp = var_wrf.sel(day=slice(date_start_ssp,date_end_ssp) )
    
    var_combine = xr.concat([var_hist,var_ssp],dim="day").rename({'day': 'time'})
    
    return (var_combine)

def extract_date_string(date_obj):
    """Extract date string from various date formats"""
    # Check if it has date attributes
    if hasattr(date_obj, 'year') and hasattr(date_obj, 'month') and hasattr(date_obj, 'day'):
        return f"{date_obj.year}-{date_obj.month:02d}-{date_obj.day:02d}"
    
    # Try to convert to integer (YYYYMMDD format)
    try:
        date_int = str(int(date_obj))
        if len(date_int) == 8:
            return f"{date_int[0:4]}-{date_int[4:6]}-{date_int[6:8]}"
    except (ValueError, TypeError):
        pass
    
    # If all else fails, try to convert to pandas datetime
    try:
        pd_date = pd.to_datetime(date_obj)
        return f"{pd_date.year}-{pd_date.month:02d}-{pd_date.day:02d}"
    except:
        raise ValueError(f"Cannot extract date from {type(date_obj)}: {date_obj}")




#WRF tier 3 reader function from stefan rahimi 
def wrfread_gcm(model,gcm,variant,mydir,var,domain,calendar):

    print(model)

    all_files = sorted(os.listdir(mydir))

    in_files = []
    for ii in all_files:
        if ii.startswith(var+".") and model in ii and gcm in ii \
            and variant in ii and domain in ii:
            if domain in ii:
                in_files.append(mydir+str(ii))

    del all_files

    nf  = len(in_files)

    data = xr.open_mfdataset(in_files, combine="by_coords")
    var_read = data.variables[var]
    day = data.variables["day"].values
    nt = len(day)
    day1_str = extract_date_string(day[0])
    day2_str = extract_date_string(day[nt-1])

    print(day1_str, day2_str)

    time_array = xr.cftime_range(start=day1_str, end=day2_str,
                              freq="1D", calendar=calendar)   

    var_read = xr.DataArray(var_read)
    var_read['day'] = time_array    #year doesn't matter here
    return (var_read)

      

def screen_times_wrf(data, date_start, date_end):
    # Dimensions should be "day"
    dask.config.set(**{"array.slicing.split_large_chunks": True})

    datedata = pd.to_datetime(data.day)
    data = data.sel(day=~((datedata.month < date_start[1]) & (datedata.year <= date_start[0])))

    datedata = pd.to_datetime(data.day)
    data = data.sel(day=~(datedata.year < date_start[0]))

    datedata = pd.to_datetime(data.day)
    data = data.sel(day=~((datedata.month >= date_end[1]) & (datedata.year >= date_end[0])))

    datedata = pd.to_datetime(data.day)
    data = data.sel(day=~(datedata.year > date_end[0]))

    return data


def get_peak_date_amt(data):
    """Get peak date and amount for each water year (Oct 1 to Oct 1)."""
    startyear = np.nanmin(data.index.year)
    endyear = np.nanmax(data.index.year)
    maxvals = []
    maxargs = []
    maxdates = []
    wys = []
    for wy in range(startyear + 1, endyear):
        startdate = datetime.date(year=wy - 1, month=10, day=1)
        enddate = datetime.date(year=wy, month=10, day=1)
        timemask = (data.index.date >= startdate) & (data.index.date < enddate)
        wydata = data.SWE[timemask]
        if len(wydata) == 0:
            continue
        maxval = np.nanmax(wydata)
        maxarg = np.nanargmax(wydata)
        maxdate = wydata.index[maxarg].date()
        maxvals.append(maxval)
        maxargs.append(maxarg)
        maxdates.append(maxdate)
        wys.append(wy)
    metrics = pd.DataFrame(
        data={"maxval": maxvals, "maxdate": maxdates, "maxarg": maxargs}, index=wys
    )
    return metrics

def metaread(dir_meta,domain):
    file = "%swrfinput_%s" %(dir_meta,domain)
    data = xr.open_dataset(file, engine ='netcdf4')
    lat = data.variables["XLAT"]
    lon = data.variables["XLONG"]
    z = data.variables["HGT"]
    return (lat,lon,z,file)
def read_wrf_meta_data(dir_meta: str, domain: str):
    """Read wrf meta data from nc4 files, and return lat, lon, height, and the filename"""
    infile = os.path.join(dir_meta, f"wrfinput_{domain}")
    data = xr.open_dataset(infile, engine="netcdf4")
    lat = data.variables["XLAT"]
    lon = data.variables["XLONG"]
    z = data.variables["HGT"]
    return (lat, lon, z, infile)

# maybe a bug in this, or in the maxarg stuff above??
def shift_to_dowy(doy):
    base = datetime.datetime(year=2002, month=1, day=1)
    targ = base + datetime.timedelta(days=doy)
    wy = 2002
    if targ.month < 10:
        wy = 2001
    wystart = datetime.datetime(year=wy, month=10, day=1)
    dowy = (targ - wystart).days
    return dowy


def relativize_snow_values(dataset):
    """
    Convert snow values in an xarray dataset to the relativized metric.

    Args:
        dataset (xarray.Dataset): Input dataset containing the snow values.

    Returns:
        xarray.Dataset: Relativized snow values dataset.
    """
    snow_values = dataset['swe'].values
    relativized_values = np.zeros((snow_values.shape))
    for i in range(9):
        for j in range(121):
            mydata = snow_values[i,j]

            # mean and standard deviation across all time steps
            mean_all = np.nanmean(mydata)
            std_all = np.nanstd(mydata)

         
            relativized_values[i,j,:,:] = (mydata - mean_all) / std_all

    relativized_dataset = dataset.copy()
    relativized_dataset['relativized_swe'] = (('gcm', 'year', 'lat2d', 'lon2d'), relativized_values)

    return relativized_dataset


def get_coords(coorddir):
    """Get the coordinates of the WRF grid and snotels."""
    #coorddir = os.path.join(projectdir, "WRF-data", "wrf_coordinates")
    dir_meta = coorddir

    # Load the location of all wrf pixels
    lat1, lon1, z1, _ = _read_wrf_meta_data(dir_meta, domain)
    lon_wrf = lon1[0, :, :]
    lat_wrf = lat1[0, :, :]
    z_wrf = z1[0, :, :]
    lat_wrf = xr.DataArray(lat_wrf, dims=["lat2d", "lon2d"])
    lon_wrf = xr.DataArray(lon_wrf, dims=["lat2d", "lon2d"])
    z_wrf = xr.DataArray(z_wrf, dims=["lat2d", "lon2d"])

    # Load the location info for all snotels
    coords = nc.Dataset(os.path.join(coorddir, "wrfinput_d02_coord.nc"))
    snoteldir = os.path.join(projectdir, "snoteldata")
    snotelmeta = pd.read_csv(os.path.join(snoteldir, "snotelmeta.csv"))
    huc6 = gpd.read_file(os.path.join(projectdir, "spatialdata", "huc6.shp"))
    huc8 = gpd.read_file(os.path.join(projectdir, "spatialdata", "huc8.shp"))
    snotel_gdf = gpd.GeoDataFrame(
        data={
            "site_name": snotelmeta.site_name,
            "elev": snotelmeta.elev,
            "site_number": snotelmeta.site_number,
            "state": snotelmeta.state,
            "namestr": snotelmeta.namestr,
            "startdt": snotelmeta.startdt,
        },
        geometry=gpd.points_from_xy(snotelmeta.lon, snotelmeta.lat),
        crs="epsg:4326",
    )

    return snotel_gdf, coords, huc6, huc8
def get_wrf_from_shp(basin, lat_wrf, lon_wrf, data_wrf):
    bounds = basin.bounds.values.flatten()
    latmask = ((lat_wrf.data >bounds[1]) & (lat_wrf.data < bounds[3]))
    lonmask = ((lon_wrf.data >bounds[0]) & (lon_wrf.data < bounds[2]))
    basinmask = (latmask & lonmask)
    pts = list(map(Point,  zip(lon_wrf[basinmask], lat_wrf[basinmask])))
    polygon = basin[0:1].geometry.values[0]
    inmask = list(map(polygon.contains,  pts))
    inmask = np.array(inmask)
    tmpdata = data_wrf[:,:,:].data.compute() ## this takes a minute... can we make it faster?
    tmpdata[:,~basinmask] = np.nan
    tmpdata2 = tmpdata[:,basinmask] 
    tmpdata2[:,~inmask] = np.nan
    return lon_wrf[basinmask], lat_wrf[basinmask], tmpdata2


def get_wrf_avail(wrfdir):
    """!
    Read in the wrf data - WIP.

    basedir [str]: base directory for wrf data, defauls to perlmutter system,
        use '/global/project/projectdirs/' for cori system
    domain [str]: domain to read in, defaults to 'd02'
    """
    # log the bc models available
    bc_glob = glob.glob(os.path.join(wrfdir, "*_bc"))
    bcmodels = [val.split("/")[-1] for val in bc_glob]
    assert len(bcmodels) > 0, f"No BC models found in {wrfdir}"
    console.log("Available BC Models:", bcmodels)
    console.log("run get_wrf_data(wrfdir,model) with the name of the model you want to load")
    return

def get_wrf_data(wrfdir, var, model, variant, domain):
    """
    TODO - fix model variable assignment using a dictionary
    """

    mod_historical = model +'_'+ variant + '_historical_bc'
    mod_future = model +'_' + variant+ '_ssp370_bc'
    gcm = mod_historical
    date_start_pd, date_end_pd = [1980, 1, 1], [2013, 12, 31]  # 30 years, historical
    model = "hist"
    modeldir = os.path.join(wrfdir, gcm , 'postprocess')
    print(modeldir)
    var_wrf = wrfread_gcm(model, gcm, variant, modeldir, var, domain)
    var_wrf = screen_times_wrf(var_wrf, date_start_pd, date_end_pd)

    # future dates
    date_start_pd, date_end_pd = [2014, 1, 1], [2100, 12, 31]
    gcm = mod_future
    modeldir = os.path.join(wrfdir, gcm ,'postprocess')
    model = "ssp370"
    var_wrf_ssp370 = wrfread_gcm(model, gcm, variant, modeldir, var, domain)
    var_wrf_ssp370 = screen_times_wrf(var_wrf_ssp370, date_start_pd, date_end_pd)

    return dict(var_wrf=var_wrf, var_wrf_ssp370=var_wrf_ssp370)


def create_wrf_df(snotel_gdf: gpd.GeoDataFrame):
    """! Create a dataframe of WRF data for each snotel site."""
    snotel_no_ak = snotel_gdf[snotel_gdf.state != "AK"]

    day1 = datetime.datetime(year=1980, day=1, month=9)
    days = []
    for i in range(42960):  # hardcoded number of days TODO fix this hardcoding
        days.append(day1 + datetime.timedelta(days=i))

    # hold the data for a dataframe
    snoteldir = os.path.join(projectdir, "snoteldata")
    entries = []
    datadir = os.path.join(projectdir, "wrfts", "ukesm1-0-ll_bc")
    for i, entry in snotel_no_ak.iterrows():
        try:
            num = entry.site_number
            name = entry.site_name.replace(" ", "").replace("(", "").replace(")", "")
            pt = [entry.geometry.x, entry.geometry.y]
            wrfpoint = np.load(os.path.join(datadir, f"wrfpoint_{name}.npy"))
            wrfbasin = np.load(os.path.join(datadir, f"wrfbasin_{name}.npy"))
            wrfpoint = pd.DataFrame(
                wrfpoint * MM_TO_IN, columns=["SWE"], index=pd.to_datetime(days)
            )
            wrfbasin = pd.DataFrame(
                wrfbasin * MM_TO_IN, columns=["SWE"], index=pd.to_datetime(days)
            )
            snotelpoint = pd.read_csv(
                os.path.join(snoteldir, f"snotel{num}.csv"),
                index_col=0,
                parse_dates=True,
            )
            wpt = get_peak_date_amt(wrfpoint)
            wbas = get_peak_date_amt(wrfbasin)
            sm = get_peak_date_amt(snotelpoint)
            entries.append(dict(name=name, wpt=wpt, wbas=wbas, sm=sm, pt=pt))
        except FileNotFoundError:
            continue
    res = gpd.GeoDataFrame(entries)
    return res



## 
def wrfread(datadir, gcm, exp, variant, domain, var):
    modeldir = datadir + gcm + '/postprocess/'+domain + '/'
    all_files = sorted(os.listdir(modeldir))
    read_files = []
    for ii in all_files:
        if (
            ii.startswith(var + ".")
            and gcm in ii
            and variant in ii
            and domain in ii
        ):
            if domain in ii:
                read_files.append(os.path.join(modeldir, str(ii)))
    assert len(read_files) > 0, f"No matching files found in {modeldir}"

    del all_files

    data = xr.open_mfdataset(read_files, combine="by_coords")
    var_read = data.variables[var]

    dates = []
    for val in data["day"].data:
        try:
            dates.append(datetime.datetime.strptime(str(val)[0:-2], "%Y%m%d").date())
        except ValueError:
            dates.append(datetime.datetime(int(str(val)[0:4]), int(str(val)[4:6]), 28))


    var_read = xr.DataArray(var_read, dims=["day", "lat2d", "lon2d"])
    var_read["day"] = dates
    return var_read

from params import gcms_dict
from dirs import savedir
def make_sddata(domain):
    gcms = gcms_dict[domain]
    sddata = {}
    for dr_t in ['all','dry','warm','warmdry','swei']:
        sddata[dr_t] = {}
        for gcm in gcms:
            sddata[dr_t][gcm] = {}

    for gcm in gcms:
        sddata[gcm] = {}
        swei = xr.open_dataset(f'{savedir}swei_max_snow_{gcm}_{domain}_BC.nc') #swei_datasets_maxsnw[f'{model}_{exp}']
        categ = xr.open_dataset(f'{savedir}categs_snow_{gcm}_{domain}_BC.nc')
        swei_cond = categ['swei_cond']
        pr_cond = categ['pr_cond']
        tas_cond = categ['tas_cond']
        sddata['all'][gcm] = swei_cond
        sddata['dry'][gcm] = swei_cond & pr_cond & ~tas_cond
        sddata['warm'][gcm] = swei_cond & ~pr_cond & tas_cond
        sddata['warmdry'][gcm] = swei_cond & pr_cond & tas_cond
        sddata['swei'][gcm] = swei 
    return sddata
    
def rescale_lists(categories):
    rescaled_categories = {}

    # Calculate the total for each year across all categories
    totals = [sum(year) for year in zip(*categories.values())]

    # Iterate over each category in the dictionary
    for category, values in categories.items():
        rescaled_values = [value / total for value, total in zip(values, totals)]
        rescaled_categories[category] = rescaled_values

    return rescaled_categories

def make_ts(domain, bounds):
    left = bounds[0]
    right = bounds[1]
    bottom = bounds[2]
    top = bounds[3]
    data2plot = make_sddata(domain)
    for key in data2plot.keys():
        for key2 in data2plot[key].keys():
            if key == 'swei':
                data2plot[key][key2] = data2plot[key][key2]['swei'][:,bottom:top,left:right]
            else:
                data2plot[key][key2] = data2plot[key][key2][:,bottom:top,left:right]
            
    #snowdata = xr.open_dataset(glob.glob(savedir + f'snow*{domain}*_BC*')[0])
    #snowarr = snowdata['snow'].data
    #snowmask = np.nansum(snowarr[0:365*10], axis = (0)) > 1500
    coords = xr.open_dataset(f'{workdir}wrfinput_{domain}')
    landmask = coords['XLAND'][0]
    landmask1 = (landmask ==1).data
    landmask1 = landmask1[bottom:top,left:right]
    maxsnow = xr.open_dataset(f'/glade/campaign/uwyo/wyom0200/berkeley/allsnowmax_BC_{domain}.nc').rename({'__xarray_dataarray_variable__':'swe'})
    if domain == 'd01': maxsnow = maxsnow.rename({'year':'time'})
    snowmask = maxsnow.sel(time = slice(maxsnow.time[0], maxsnow.time[20])).mean(dim = 'time').mean(dim = 'gcm').swe > 50
    snowmask = snowmask.rename({'lat2d':'lat', 'lon2d':'lon'})
    snowmask = snowmask[bottom:top,left:right]
    
    gcms = data2plot['all'].keys()
    
    categories = {
        'dry': None,
        'warmdry': None,
        'warm': None,}
    for name in categories.keys():
        tmp = []
        for gcm in gcms:
            tmp.append(data2plot[name][gcm].where(landmask1).where(snowmask))
        ds_concat = np.array(tmp)
        categories[name] = np.nanmean(ds_concat, axis = (0,2,3))
    wet = [] 
    lownorm = []
    highnorm = []
    for gcm in gcms:
        lownorm.append(((data2plot['swei'][gcm] >= -0.8) & (data2plot['swei'][gcm] < 0 )).where(landmask1).where(snowmask))
        highnorm.append(((data2plot['swei'][gcm] >= 0) & (data2plot['swei'][gcm] < 0.8 )).where(landmask1).where(snowmask))
        wet.append((data2plot['swei'][gcm] >= 0.8).where(landmask1).where(snowmask))
    wet2 =  np.nanmean(np.array(wet), axis = (0,2,3))

    categories['lownorm'] = np.nanmean(np.array(lownorm), axis = (0,2,3)) #  1 - categories['dry'] - categories['warm'] - categories['warmdry'] - wet2
    categories['highnorm'] = np.nanmean(np.array(highnorm), axis = (0,2,3))
    categories['wet'] = wet2

        
    return categories