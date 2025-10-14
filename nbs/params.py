## define global parameters ##
import xarray as xr
from dirs import basedir, savedir, wrfdir 


## names of all models to query
allnames = [ 'ukesm1-0-ll_bc']
## which experiment ids?
experiment_ids = ['historical', 'ssp370']
## which variables? 
variables = ['snow','t2','prec']

domains = ['d02'] # , 'd02', 'd03', 'd04']
coordsdict = {}
for domain in domains:
    coordsdict[domain] = xr.open_dataset(f'/glade/work/mcowherd/wrfinput_{domain}')
    
years = {'historical': [1850,2014],
         'ssp370':[2015,2099],
         'ssp585':[2015,2099]}

table_ids = {'tas':'Amon', 
             'pr': 'Amon',
             'snw':'LImon'}
labels =  {'snw':'SWE [kg m$^{-2}$]', 
             'pr': 'Precip [kg $d^{-1}$]',
             'tas':' Temp [K]'}

colors = {'historical':'black',
          'ssp245':'darkblue',
          'ssp585':'darkred'}

global_params = {'vars': ['prec','t2','snow'],
          'states' : ['CA', 'WY'],
          'res': ['45','9','3']}
res_domain = {'3': ['d03','d04'],
               '9' :['d02'],
               '45':['d01']}
domain_state = {'d01':['CA','WY'],
                'd02':['CA','WY'],
                'd03':['CA'],
                'd04':['WY'],}
boundaries = {'d01':{'CA': [43,60,35,68],
                     'WY': [68,81,43,63]},
              'd02':{'CA': [44,125,87,251],
                     'WY': [165,230,130,227]},
              'd03':{'CA': [0,-1,0,-1]},
              'd04':{'WY': [0,-1,0,-1]}}
state_res_domain = {'CA': {'45':'d01',
                            '9': 'd02',
                            '3':'d03'},
                     'WY': {'45':'d01',
                            '9': 'd02',
                            '3':'d04'},}
gcms_dict = {'d01': ['cesm2','mpi-esm1-2-lr','cnrm-esm2-1',
                     'ec-earth3-veg','fgoals-g3','ukesm1-0-ll',
                     'canesm5','access-cm2','ec-earth3'],
             'd02': ['cesm2','mpi-esm1-2-lr','cnrm-esm2-1',
                     'ec-earth3-veg','fgoals-g3','ukesm1-0-ll',
                     'canesm5','access-cm2','ec-earth3'],
             'd03': ['ec-earth3-veg'],
             'd04': ['ec-earth3-veg'],}

variants = {'cesm2':'r11i1p1f1',
            'mpi-esm1-2-lr':'r7i1p1f1',
            'cnrm-esm2-1': 'r1i1p1f2',
            'ec-earth3-veg': 'r1i1p1f1',
            'fgoals-g3': 'r1i1p1f1',
            'ukesm1-0-ll': 'r2i1p1f2',
            'canesm5': 'r1i1p2f1',
            'access-cm2': ' r5i1p1f1',
            'ec-earth3': 'r1i1p1f1'}
bc = 'BC'

variants_dict = {'d01': ['r11i1p1f1','r7i1p1f1','r1i1p1f2',
                         'r1i1p1f1','r1i1p1f1','r2i1p1f2',
                         'r1i1p2f1','r5i1p1f1','r1i1p1f1',],
                 'd02': ['r11i1p1f1','r7i1p1f1','r1i1p1f2',
                         'r1i1p1f1','r1i1p1f1','r2i1p1f2',
                         'r1i1p2f1','r5i1p1f1','r1i1p1f1',],
                 'd03': ['r1i1p1f1'],
                 'd04': ['r1i1p1f1'],}

calendars_dict = {'d01': ['365_day','proleptic_gregorian','proleptic_gregorian',
                         'proleptic_gregorian','365_day','360_day',
                         '365_day','proleptic_gregorian','proleptic_gregorian',],
                 'd02': ['365_day','proleptic_gregorian','proleptic_gregorian',
                         'proleptic_gregorian','365_day','360_day',
                         '365_day','proleptic_gregorian','proleptic_gregorian',],
                 'd03': ['proleptic_gregorian'],
                 'd04': ['proleptic_gregorian'],}

ssps_dict = {'d01': ['ssp370','ssp370','ssp370','ssp370',
                     'ssp370','ssp370','ssp370','ssp370',
                     'ssp370',],
             'd02': ['ssp370','ssp370','ssp370','ssp370',
                     'ssp370','ssp370','ssp370','ssp370',
                     'ssp370',],
             'd03': ['ssp370'],
             'd04': ['ssp370'],}


source_ids = {'access-cm2':'ACCESS-CM2', 
             'cesm2':'CESM2',
             'cnrm-esm2-1':'CNRM-ESM2-1', 
             'canesm5':'CanESM5',
             'ec-earth3':'EC-Earth3',
             'ec-earth3-veg': 'EC-Earth3-Veg',
             'fgoals-g3':'FGOALS-g3', 
             'mpi-esm1-2-lr':'MPI-ESM1-2-LR',
             'ukesm1-0-ll':'UKESM1-0-LL'}

date_start_hist, date_end_hist = "1980-09-01", "2014-08-31"
date_start_ssp, date_end_ssp = "2014-09-01", "2100-08-31"
