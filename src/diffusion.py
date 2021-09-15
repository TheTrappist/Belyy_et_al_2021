# -*- coding: utf-8 -*-
"""

This module contains a collection of functions for analyzing diffusion data.

Written by Vladislav Belyy (UCSF)

"""
import os
import numpy as np
import math
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pylab as plt


### Definitions of short useful functions #####################################
def ax_fun(x, a):
    # Line with zero y-intercept
    return a*x

def get_residuals(data_x, data_y, slope, y_intercept):
    # Calculate residuals of a linear fit for x-y data
    model_y = data_x * slope + y_intercept
    resid = data_y - model_y
    return resid

def get_sum_squares(data):
    # Calculate total sum of squares
    mean = np.mean(data)
    sum_sq = np.sum(np.square(data - mean))
    return sum_sq

def get_r_sq(data_x, data_y, slope, yint):
    # Calculate r-squared statistic for a given line
    resid = get_residuals(data_x, data_y, slope, yint)
    sum_resid = np.sum(np.square(resid))
    sum_sq = get_sum_squares(data_y)
    r_sq = 1 - sum_resid/sum_sq
    return r_sq

### Definitions of major functions ############################################
    
def diffconst_gw(R, T, c, eta=0.056):
    """Calculate diffusion constant based on Guigas and Weiss 2006 paper.
    
    Returns the calculated diffusion constant of a membrane inclusion, as
    derived in the 2006 paper by Guigas and Weiss, 
    doi 10.1529/biophysj.106.087031 .
    
    Args:
        R (float): radius of inclusion, in nm.
        T (float): Temperature, in Kelvin.
        c (float): boundary conditions, in nm (c=6 for stick, c=4 for slip).
        eta (float): membrane viscosity, in Pa s. Defaults to 0.056 (the value
            used in the paper).
        
    Returns:
        D (float): Diffusion constant, in um^2/s.
 
    """
    
    k = 1.38e-23 # Boltzmann constant in J*K^(-1)
    R_m = R / 1e9 # convert nm to meters
    u_conv = 1e12 # to convert from m^2/s to um^2/s
    D = u_conv*k*T*math.atan(c/float(R)) / (8*np.pi *eta *R_m)
    return D

def diffconst_sd(R, T, c, eta_m=0.19, eta_c=0.039, h=3.5):
    """Calculate diffusion constant based on work by Saffman and Delbruck.
    
    Returns the calculated diffusion constant of a membrane inclusion, as
    derived in the 1975 PNAS paper, doi 10.1073/pnas.72.8.3111 .
    
    Args:
        R (float): radius of inclusion, in nm.
        T (float): Temperature, in Kelvin.
        c (float): boundary conditions, in nm (c=6 for stick, c=4 for slip).
        eta_m (float): membrane viscosity, in Pa s. Defaults to 0.19 (value
            used in the 2006 Guigas and Weiss paper).
        eta_c (float): solvent viscosity, in Pa s. Defaults to 0.039 (value
            used in the 2006 Guigas and Weiss paper).
        h (float): thickness of membrane, in nm. Defaults to 3.5 (value
            used in the 2006 Guigas and Weiss paper).
        
    Returns:
        D (float): Diffusion constant, in um^2/s.
 
    """
    
    k = 1.38e-23 # Boltzmann constant in J*K^(-1)
    gamma = 0.5772 # Euler's constant
    
    m2s_um2s = 1e12 # to convert from m^2/s to um^2/s
    nm_m = 1e-9 # to convert from nm to m
    
    h_m = h * nm_m # height in meters
    
    D = m2s_um2s * k*T*(np.log(h*eta_m/(R*eta_c))-gamma) / (4*np.pi*eta_m*h_m)
    
    return D

def calc_msd (track_coords, time_step):
    """Calculate and return mean squared displacement from provided tracks.
    
    Args:
        track_coords (list): numpy arrays of x,y,t spot positions, one array for
           each track.
        time_step (float): duration, in seconds, of each frame of the movie.
       
    Returns:
        t_dsq (list): numpy arrays of time vs. squared displacement, by track.
        msd_data (tuple): Contains the following data.
            time (numpy array): time vector for MSD plot
            msd (numpy array): mean squared displacements
            std (numpy array): standard deviations of MSD
            sterr (numpy array): standard errors of MSD 
    """
    t_dsq = list() # time vs sq. displacement for each particle
    for track in track_coords:
        # 'track' is an array of x,y,t

        track_norm = track - track[0,:] # set x,y,t=0
        xy_sq = np.square(track_norm[:,0:2])
        d_sq = np.sum(xy_sq, axis=1)
        t = track_norm[:,2] * time_step
        tds = np.vstack((t, d_sq))
        tds = np.transpose(tds)

        t_dsq.append(tds)

    # rebuild results as a padded numpy array for calculating means
    l_max = max((len(el) for el in t_dsq))

    d_sq = np.empty((l_max, len(t_dsq)))
    t_total = np.empty((l_max, len(t_dsq)))
    for i, x in enumerate(t_dsq):
        len1 = x.shape[0]
        pad_len = l_max - len1
        nanpad = np.zeros([pad_len,2])*np.nan
        x_padded = np.concatenate((x,nanpad), axis=0)

        d_sq[:,i] = x_padded[:,1]
        t_total[:,i] = x_padded[:,0]

    time = np.nanmean(t_total, axis=1)
    msd = np.nanmean(d_sq, axis=1)
    std_dsq = np.nanstd(d_sq, axis=1)
    sterr_dsq = stats.sem(d_sq, axis=1, nan_policy='omit',ddof=0)
    
    msd_data = (time, msd, std_dsq, sterr_dsq)
    
    return t_dsq, msd_data

def fit_diffusion_const(msd_data, dim=2, nframes=10):
    """Fit MSD data to a simple linear regression.
    
    Args:
        msd_data (tuple): Contains the following data.
            time (numpy array): time vector for MSD plot
            msd (numpy array): mean squared displacements
            std (numpy array): standard deviations of MSD
            sterr (numpy array): standard errors of MSD
        dim (int): dimensionality of the diffusion. Defaults to 2.
        nframes (int): number of frames from the start of the movie to be used
            for fitting. Defaults to 10.
    
    Returns:
        fit_params (dict): parameters of the fit.        
    """
    time, mean_dsq, __, __ = msd_data
    t_temp = time[0:nframes]
    msd_temp = mean_dsq[0:nframes]
    # get the slope:
    popt, pcov = curve_fit(ax_fun, t_temp, msd_temp)
    slope = popt[0]
    stdev = np.sqrt(np.diag(pcov))
    #slope = np.dot(t_temp, msd_temp) / np.dot(t_temp, t_temp)
    dc = slope / (2 * dim) # diffusion constant
    fit_params = {}
    fit_params['slope'] = slope
    fit_params['dc'] = dc
    fit_params['stdev'] = stdev
    
    return fit_params

def plot_msd(t_dsq, msd_data, fit_params=None, rc_params=[18,8], ms=3):
    """Plot the results of an MSD calculation and individual trajectories.
    
    Args:
        t_dsq (list): numpy arrays of time vs. squared displacement, by track.
        msd_data (tuple): Contains the following data.
            time (numpy array): time vector for MSD plot
            msd (numpy array): mean squared displacements
            std (numpy array): standard deviations of MSD
            sterr (numpy array): standard errors of MSD
        fit_params (dict): parameters of fit to be plotted, if desired
        rc_params (list): matplotlib figsize setting. Defaults to [18,8].
        ms (float): marker size for the scatterplot. Defaults to 3.

    Returns:
        fig (matplotib figue) : Output figure
        axarr (matplotlib axes array): Axes of the output figure (2 elements)          
    """
    plt.rcParams["figure.figsize"] = rc_params
    
    # determine whether there are multiple trajectories present
    if len(t_dsq) > 1:
        mult_traj = True
    else:
        mult_traj = False
    
    time, mean_dsq, std_dsq, sterr_dsq = msd_data
    
    fig, axarr = plt.subplots(1,2)
    
    if mult_traj:
        axarr[0].set_title('All track trajectories')
    else:
        axarr[0].set_title('Particle trajectory')
    axarr[0].set_xlabel('Time (s)')
    
    axarr[0].set_ylabel(r'Squared displacement ($\mu m^2$)')
    axarr[1].set_title('Mean squared displacement')
    axarr[1].set_xlabel('Time (s)')
    axarr[1].set_ylabel(r'MSD ($\mu m^2$)')
    
    # Plot individual trajectories
    for x in t_dsq:
        axarr[0].plot(x[:,0], x[:,1])
        
    # Plot MSD with errors
    axarr[1].errorbar(time, mean_dsq, yerr=sterr_dsq, fmt='o', ms=ms)
    
    if fit_params is not None:
        label = r'D = {0:.5f} $\mu m^2/s$'.format(fit_params['dc'])     
        t_fit = np.arange(0,(max(time)*1.01), max(time)/5)
        y_fit = t_fit * fit_params['slope']
        axarr[1].plot(t_fit,y_fit,label=label)
        axarr[1].legend(loc='upper left', shadow=True)

    
    return fig, axarr

def bootstrap_linreg (x, y, slope=None, nreps=1000, ci=95):
    """Calculate bootstrap confidence intervals for a linear regression.
    
    Fits the provided data to a straight line (optionally with a fixed slope),
    and calculates percentile confidence intervals for the linear fit and the
    goodness of fit (R-squared) parameter by subsampling from the original 
    dataset.
    
    
    Args:
        x (array_like): x-coordinates of the data.
        y (array_like): y-coordinates of the data. Must have the same legnth as
            x.
        slope (float): Fixed slope for the linear fit. If None, this function
            performs a true linear regression  and finds both slope and 
            y-intercept. Defaults to None.
        nreps (int): Number of bootstrap iterations. Defaults to 1000.
        ci (float): confidence interval for bootstrap results. Defaults to 95.
            
    Returns:
         result (dict): results of the bootstrap analysis.
    """

    len_data = len(x)
    indices = np.arange(len_data)
    
    slopes = np.zeros(nreps)
    y_ints = np.zeros(nreps)
    r_sqs = np.zeros(nreps)
    
    # Perform the bootstrap
    for i in range(nreps):
        # Select samples with replacement
        subsamp_indices = np.random.choice(indices, len_data)
        subsamp_x = x[subsamp_indices]
        subsamp_y = y[subsamp_indices]
        
        # Fit line (y = ax + b) to the selected samples
        if slope: # fixed slope
            a = slope
            b = np.mean(subsamp_y - a * subsamp_x)
        else: # true regression
            r = stats.linregress(subsamp_x,subsamp_y)
            a = r.slope
            b = r.intercept
            
        r_sq = get_r_sq(subsamp_x,subsamp_y,a,b)
        
        slopes[i] = a
        y_ints[i] = b
        r_sqs[i] = r_sq
      
    # Calculate means and confidence intervals
    mean_slope = np.mean(slopes)
    mean_y_int = np.mean(y_ints)
    mean_r_sqs = np.mean(r_sqs)
    
    ci_low_high = [50 - ci/2, 50 + ci/2]
    ci_slope = np.percentile(slopes, ci_low_high)
    ci_int = np.percentile(y_ints, ci_low_high)
    ci_rsq = np.percentile(r_sqs, ci_low_high)
    
    result = {"slope" : mean_slope,
              "y_int" : mean_y_int,
              "r_sq" : mean_r_sqs,
              "ci_slope" : ci_slope,
              "ci_y_int" : ci_int,
              "ci_rsq" : ci_rsq}
    return result

def make_fig_save_dirs(save_dir_root, pdf=False):
    """Create directories for saving figures in svg and png formats
    
    Args:
        save_dir_root (string): root directory for saving figures

    Returns:
        dir_svg (string): valid directory
        dir_png (string): valid directory          
    """
    dir_png = os.path.join(save_dir_root, 'png')
    dir_svg = os.path.join(save_dir_root, 'svg')
    if not os.path.exists(dir_png):
        os.makedirs(dir_png)
    if not os.path.exists(dir_svg):
        os.makedirs(dir_svg)
        
    return dir_svg, dir_png

def save_fig(fig, file_name, dir_svg, dir_png):
    """Save figure in svg and png formats.
    
    Args:
        fig (matplotlib figure): figure to be saved
        file_name (string): desired name of figure file
        dir_svg (string): valid directory
        dir_png (string): valid directory  

    Returns:
        None         
    """
    fig_filename_svg = os.path.join(dir_svg, (file_name+'.svg'))
    fig_filename_png = os.path.join(dir_png, (file_name+'.png'))
    fig.savefig(fig_filename_svg)
    fig.savefig(fig_filename_png)
    
    return

def save_fig_pdf(fig, file_name, dir_pdf):
    """Save figure in pdf format.
    
    Args:
        fig (matplotlib figure): figure to be saved
        file_name (string): desired name of figure file
        dir_pdf (string): valid directory

    Returns:
        None         
    """
    fig_filename_pdf = os.path.join(dir_pdf, (file_name+'.pdf'))
    fig.savefig(fig_filename_pdf)
    
    return