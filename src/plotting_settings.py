# -*- coding: utf-8 -*-
"""

This is a short script for setting default plotting parameters so that figures
appear consistent across Jupyter notebooks. Call this script with a single
argument that can have the value 'save', 'save_large', or 'plot_only', which 
switches between figure export and plotting-only pre-sets.

Written by Vladislav Belyy (UCSF).

"""
import matplotlib
import matplotlib.pylab as plt
import sys

mode = sys.argv[1]

if mode == 'save':
    matplotlib.rcParams['figure.figsize'] = 1.6, 1.4

    # Set up fonts
    matplotlib.rc("font", family="Arial")

    matplotlib.rcParams['pdf.fonttype'] = 42 # Make fonts editable
    matplotlib.rcParams['axes.linewidth']= 0.5
    matplotlib.rcParams['lines.linewidth'] = 0.5

    SMALL_SIZE = 5
    MEDIUM_SIZE = 6
    BIGGER_SIZE = 7

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

elif mode == 'save_large': # Full-page figure
    matplotlib.rcParams['figure.figsize'] = 7, 3.5
    matplotlib.rcParams['figure.autolayout'] = True # Auto-adjust fig spacing

    # Set up fonts
    matplotlib.rc("font", family="Arial")

    matplotlib.rcParams['pdf.fonttype'] = 42 # Make fonts editable
    matplotlib.rcParams['axes.linewidth']= 1
    matplotlib.rcParams['lines.linewidth'] = 1

    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

else:
    matplotlib.rcParams['figure.figsize'] = 6, 4
