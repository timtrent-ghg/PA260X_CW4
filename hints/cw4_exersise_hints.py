import matplotlib.pyplot as plt
import numpy as np
import netCDF4
from scipy import stats

##################################################################################################
# here are all the functions
##################################################################################################
def read_csv(filename):
    """ read in the csv files we previously made of UTH anomaly time series.
    inputs: filename -> full path to file containing our data

    ouputs: results -> python dictionary containing our anomaly time series.
    """
    results = {}
    data = np.loadtxt(filename, skiprows=1, delimiter=",",dtype='<f4')
    header = np.genfromtxt(filename, delimiter=',', dtype=str, max_rows=1)
    for ii, varname in enumerate(header):
        results[varname] = data[:,ii]
    return results
    
##################################################################################################
def simple_moving_average(yvals, width):
    """ compute the moving average of a time series with a user defined sliding window.
    inputs: yvals -> time series on regular time steps
            width -> width of sliding window (n time steps)
    output: sommothed time series
    """
    return np.convolve(yvals, np.ones(width), 'same')/width

##################################################################################################
def estimate_coef(x, y):
    """ python implimentation of linear regression.
    Inputs: x -> 1D array (e.g. fractional year)
            y -> 1D array (e.g. UTH anomaly)

    Outputs: coeffs -> tuple containing the intercept and gradient
    """
    # define number of observations/points
    n = np.size(x)
    
    # calculate the mean of the x and y vectors
    m_x = np.mean(x)
    m_y = np.mean(y)
    
    # calculate cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    
    # calculate regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return (b_0, b_1)
    
##################################################################################################
def read_gridded_data(filename, report=True):
    """ function to read regular gridded data files and return a Python dictionary.
    inputs: filename -> name of file to be read
            report   -> boolean flag, if true then a table listing the file contents is printed to screen

    outputs: data    -> a python dictionary containing the file contents
    """
    # open the file and map the contents to a netCDF object. Use a test to capture
    # any issues with the data file
    try:
        nc = netCDF4.Dataset(filename,"r")
    except IOError:
        raise
    # if report is set to true then prind global attributes
    if report == True:
        print(f"Global Attributes For File: {filename}") # when witing strings, starting them with an 'f' allows you to insert other variables using {}.
        print("----------------------------------------------------------------------------------------------------")
        print(nc)
    # Define a dictionary to hold the contents
    data = {} 
    # loop over the file contents and write
    if report == True:
        # what variables are in this file?
        # printing Aligned Header 
        print("-------------------------------------------------------------------")        
        print(f"{'Variable Name' : <14} |{'Long Name':<32} |{'tdim, ydim, xdim':>17}") 
        print("-------------------------------------------------------------------")
    else:
        pass # we add a pass so we can close this if statement with an else

    for varname in nc.variables.keys():
        if report == True:
            print(f"{varname:<14} |{nc[varname].long_name:<32} |{', '.join([str(d) for d in nc[varname].shape]):>17}")
        else:
            pass
        # write the variable to the dictionary
        data[varname] = nc[varname][:] #  the [:] at the end indexes all the data
    
    # finally close the file
    nc.close()

    # return the filled data dictionary
    return data
    
##################################################################################################
# Here are all the steps in the code
##################################################################################################
# read in the data
data = read_csv("data/uth_anomoly_timeseries.csv")

##################################################################################################
# apply a simple 12 month moving average filter to our time series.
anom_sma_glb = simple_moving_average(data['Global'],12)
anom_sma_lnd = simple_moving_average(data['Land'],12)
anom_sma_ocn = simple_moving_average(data['Ocean'],12)

###################################################################################################
# regress smoothed time series against fractional year and return linear fit coefficients
glb_coeffs = estimate_coef(data['Frac_Year'], anom_sma_glb)
lnd_coeffs = estimate_coef(data['Frac_Year'], anom_sma_lnd)
ocn_coeffs = estimate_coef(data['Frac_Year'], anom_sma_ocn)

###################################################################################################
# plot the UTH timeseries
units=r" %/dec" 
fig = plt.figure(num=1,figsize=(10,4),dpi=300)  
plt.plot(data['Frac_Year'], anom_sma_glb,lw=2,color='#1F3649',label=f"Global ({glb_coeffs[1]*120:0.2f}{units})")
yvalsg = glb_coeffs[0]+glb_coeffs[1]*data['Frac_Year']
plt.plot(data['Frac_Year'], yvalsg, '--',color='#1F3649')

plt.plot(data['Frac_Year'], anom_sma_lnd,lw=2,color="#2E75B6",label=f"Land ({lnd_coeffs[1]*120:0.2f}{units})")
yvalsl = lnd_coeffs[0]+lnd_coeffs[1]*data['Frac_Year']
plt.plot(data['Frac_Year'], yvalsl, '--',color="#2E75B6")

plt.plot(data['Frac_Year'], anom_sma_ocn,lw=2,color="#C55A11",label=f"Ocean ({ocn_coeffs[1]*120:0.2f}{units})")
yvalso = ocn_coeffs[0]+ocn_coeffs[1]*data['Frac_Year']
plt.plot(data['Frac_Year'], yvalso, '--',color="#C55A11")
         
plt.xlim(1979,2025)
plt.ylim(-2.5,1.5)
plt.ylabel(r"$\Delta$UTH (%)")
plt.xlabel("year")
plt.title(r"HIRS Clear Sky UTH 1979-2024 ($\pm$30$^{\circ}$)")
plt.legend(loc=3)

plt.tight_layout()

###################################################################################################
## Task 2:
# step 1: read in the data
era5_lsm_data = read_gridded_data("data/pa260x_ecmwf_Land_sea_mask_2.5_x_2.5_1940_2024.nc",report=False)
era5_skt_data = read_gridded_data("data/pa260x_ecmwf_Skin_temperature_2.5_x_2.5_1940_2024.nc",report=False)

###################################################################################################
# step 2: select data between +/-30, note here we use a where statement
# with two conditions that need to be true so we use '&'
find = np.where((era5_lsm_data['latitude'][:] >= -30)&(era5_lsm_data['latitude'][:] <= 30))

# lets test with the land sea mask
plt.figure(figsize=(12,2),dpi=200)
plt.pcolormesh(era5_lsm_data['longitude'], era5_lsm_data['latitude'][find], era5_lsm_data['lsm'][find[0],:],vmin=0,vmax=1)
plt.colorbar()

###################################################################################################
# step 3: Now lets select our data from the two python dictoinaries so it is for our region of interest (ROI)
lons = era5_lsm_data['longitude']
lats = era5_lsm_data['latitude'][find]
lsm = era5_lsm_data['lsm'][find[0],:]
skt = era5_skt_data['skt'][:,find[0],:]
# we can use use it to create a few masks for seperating land and ocean grid cells
land = np.where(lsm == 1)
ocean = np.where(lsm == 0)

###################################################################################################
# step 4: calculate the weights neededin our time series calculation
# use meshgrid to create a 2D array of latitudes - we also get longitudes at the same time
x2d, y2d = np.meshgrid(lons,lats )

# calculate the weights
wgts = np.cos(np.radians(y2d))

# delete the data we now longer need
del y2d,x2d


###################################################################################################
# step 5: define our arrays tohold the weighted mean skin temperature time series for each case
tdim = era5_skt_data['time'].size
skt_ts_glb = np.full(tdim, np.nan)
skt_ts_lnd = np.full(tdim, np.nan)
skt_ts_ocn = np.full(tdim, np.nan)

# loop over each time step and andcalculated weighted mean UTH. In the case of the land and
# ocean vaues we apply the masks calculated previously
for tt, tmp in enumerate(skt):
    mdx = tt % 12
    skt_ts_glb[tt] = np.nansum(tmp*wgts)/np.nansum(wgts)
    skt_ts_lnd[tt] = np.nansum(tmp[land]*wgts[land])/np.nansum(wgts[land])
    skt_ts_ocn[tt] = np.nansum(tmp[ocean]*wgts[ocean])/np.nansum(wgts[ocean])

# make a simple plot of the results
plt.figure(figsize=(8,3),dpi=300)
plt.plot(era5_skt_data['time'], skt_ts_glb, label='Global')
plt.plot(era5_skt_data['time'], skt_ts_lnd, label='Land')
plt.plot(era5_skt_data['time'], skt_ts_ocn, label='Ocean')
plt.xlim(1940,2025)
plt.legend(loc=4)

###################################################################################################
# step 6: calculate the anomaly time series and plot
# define ourr arrays
anom_glb = np.full(tdim, np.nan)
anom_lnd = np.full(tdim, np.nan)
anom_ocn = np.full(tdim, np.nan)

# we will use a 20 year reference period between 1980 to 2000
ref_time = np.where((era5_skt_data['time'] >= 1980)&(era5_skt_data['time'] < 2001))

# loop over each month step between 0-11 and calculate the anomaly
for mth in range(12):
    anom_glb[mth::12] = skt_ts_glb[mth::12] - np.nanmean(skt_ts_glb[ref_time][mth::12])
    anom_lnd[mth::12] = skt_ts_lnd[mth::12] - np.nanmean(skt_ts_lnd[ref_time][mth::12])
    anom_ocn[mth::12] = skt_ts_ocn[mth::12] - np.nanmean(skt_ts_ocn[ref_time][mth::12])

# make a simple plot
plt.figure(figsize=(8,3),dpi=300)
plt.plot(era5_skt_data['time'], anom_glb, label='Global')
plt.plot(era5_skt_data['time'], anom_lnd, label='Land')
plt.plot(era5_skt_data['time'], anom_ocn, label='Ocean')
plt.xlim(1940,2025)
plt.legend(loc=4)

##################################################################################################
# This our code to save the outputs
# we open a file called uth_anomoly_timeseries.csv and assign it to a file object we have named fobj
# we can the loop over each time step and write the results to file
with open("skt_anomoly_timeseries_pm30_deg.csv", "w") as fobj:
    # first we write a header, the \n tells the computer to start a new line
    fobj.write("Frac_Year,Global,Land,Ocean\n")
    # here we use zip in order to interate over all arrays at the same time
    for yr, v1, v2, v3 in zip(era5_skt_data['time'], anom_glb,anom_lnd,anom_ocn):
        # we create a f string and write it to file
        fobj.write(f"{yr:0.3f}, {v1:0.2f}, {v2:0.2f}, {v3:0.2f}\n")

##################################################################################################
# step7: apply a simple 12 month moving average filter to our time series.
skt_anom_sma_glb = simple_moving_average(anom_glb,12)
skt_anom_sma_lnd = simple_moving_average(anom_lnd,12)
skt_anom_sma_ocn = simple_moving_average(anom_ocn,12)

##################################################################################################
# step 7: select the same time period for analysis 
mkr = np.where((era5_skt_data['time'] >= 1979)&(era5_skt_data['time'] <= 2025))

# regress smoothed time series against fractional year and return linear fit coefficients
g_coeffs = estimate_coef(skt_anom_sma_glb[mkr], anom_sma_glb)
l_coeffs = estimate_coef(skt_anom_sma_lnd[mkr], anom_sma_lnd)
o_coeffs = estimate_coef(skt_anom_sma_ocn[mkr], anom_sma_ocn)

# calculate the pearson coeeficients
rg = stats.pearsonr(skt_anom_sma_ocn[mkr], anom_sma_ocn)
rl = stats.pearsonr(skt_anom_sma_lnd[mkr], anom_sma_lnd)
ro = stats.pearsonr(skt_anom_sma_glb[mkr], anom_sma_glb)

# convert pearson coefficient to a string and add a * if significant
if rg[1] < 0.05:
    rgs = f"{rg[0]:0.2f}*"
else:
    rgs = f"{rg[0]:0.2f}"

if rl[1] < 0.05:
    rls = f"{rl[0]:0.2f}*"
else:
    rls = f"{rl[0]:0.2f}"

if ro[1] < 0.05:
    ros = f"{ro[0]:0.2f}*"
else:
    ros = f"{ro[0]:0.2f}"

# make a simple set of scatter plots of skin temp vs UTH
plt.figure(figsize=(9,3),dpi=200)
plt.subplot(131)
plt.plot(skt_anom_sma_glb[mkr], anom_sma_glb, 'o',label=f"{g_coeffs[0]:0.2f} %/K")
plt.ylabel(r"$\Delta$UTH (%)")
plt.xlabel(r"$\Delta$T$_{s}$ (K)")
plt.title(f'Global (r={rgs})')
plt.legend()

plt.subplot(132)
plt.plot(skt_anom_sma_ocn[mkr], anom_sma_ocn, 'o',label=f"{o_coeffs[0]:0.2f} %/K")
plt.ylabel(r"$\Delta$UTH (%)")
plt.xlabel(r"$\Delta$T$_{s}$ (K)")
plt.title(f'Ocean (r={ros})')
plt.legend()

plt.subplot(133)
plt.plot(skt_anom_sma_lnd[mkr], anom_sma_lnd, 'o',label=f"{l_coeffs[0]:0.2f} %/K")
plt.ylabel(r"$\Delta$UTH (%)")
plt.xlabel(r"$\Delta$T$_{s}$ (K)")
plt.title(f'Land (r={rls})')
plt.legend()
plt.tight_layout()

##################################################################################################
# step 8: Calculate ENSO anomolies and test their correlation against UTH anomolies
# load the functions from enso_tools.ipynb
%run enso_tools

# calculate the SST anaomolies for ENSO regions
sst_anoms = create_enso_timeseries(skt, lats, lons, lsm)

# calculate pearson corrleations and print a simple table to screen
calc_peason_r(sst_anoms, anom_sma_ocn, anom_sma_lnd, anom_sma_glb, mkr)