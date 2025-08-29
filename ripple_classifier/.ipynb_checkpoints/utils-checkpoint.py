"""
Utils Functions used for ripple type classification.
Created on 29th August 2025

Author: Manfredi Castelli - manfredi.castelli98@gmail.com
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import seaborn as sns


sr = 1250. # sampling rate
DTT = 1000/sr # sampling time (in ms) of eeg signal

# Plot style fonts
fontdict = {
        'style':'normal', #‘normal’, ‘italic’, ‘oblique’
        'color':  'k',
        'weight': 'normal',
        'size': 18,
        }
col_paired = sns.color_palette('Paired',20)

def get_root_directory():
    from pathlib import Path
    root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    return root

def lowpass(signal, fcut, sr=1250., FilterOrder=4, axis=-1):
    '''Low Pass Butterworth filter using cut-off frequency fcut'''
    lof_ = (2. / sr) * fcut  # Hz * (2/SamplingRate)

    # filtering
    b, a = sig.butter(FilterOrder, lof_, 'low')  # filter design
    output = sig.filtfilt(b, a, signal, axis=axis)

    return output

    
def MatrixGaussianSmooth(Matrix,GaussianStd,GaussianNPoints=0,NormOperator=np.sum):

	# Matrix: matrix to smooth (rows will be smoothed)
	# GaussiaStd: standard deviation of Gaussian kernell (unit has to be number of samples)
	# GaussianNPoints: number of points of kernell
	# NormOperator: # defines how to normalise kernell

	if GaussianNPoints<GaussianStd:
		GaussianNPoints = int(4*GaussianStd)

	GaussianKernel = sig.get_window(('gaussian',GaussianStd),GaussianNPoints)
	GaussianKernel = GaussianKernel/NormOperator(GaussianKernel)

	if len(np.shape(Matrix))<2:
		SmoothedMatrix = np.convolve(Matrix,GaussianKernel,'same')
	else:
	    	SmoothedMatrix = np.ones(np.shape(Matrix))*np.nan
	    	for row_i in range(len(Matrix)):
	    		SmoothedMatrix[row_i,:] = \
		    		np.convolve(Matrix[row_i,:],GaussianKernel,'same')

	return SmoothedMatrix,GaussianKernel


def runCSD_(LFPs):
        nChs = np.size(LFPs,0)
        nSamples = np.size(LFPs,1)

        CSD = np.zeros((nChs,nSamples))
        for chi in range(1,nChs-1):
                CSD[chi,:] = -(LFPs[chi-1,:]-2*LFPs[chi,:]+LFPs[chi+1,:])
                
        return CSD

    
def groupConsec(data,minsize=1):
    '''Group consecutive values'''
    from itertools import groupby
    from operator import itemgetter

    groups =[]

    for k,g in groupby(enumerate(data),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))

        if np.size(group)>=minsize:
                groups.append(np.array(group))

    return groups 

def runCSD(LFPs,smooth=True,spacing=20,sm=50):
    '''
    Returns CSD for signals in LFPs.

    CSD = runCSD(LFPs)

    input: LFPs - array (channels, time samples)
                IMPORTANT!: remember to order channels in LFPs according to their spatial locations.
                Typically, LFPs are actually event-triggered averages, but you can also input raw traces.
                Event-triggers might be troughs/peaks of an oscillation.
                If you are interested in the CSD of a particular oscillation you might input filtered traces.

    output: CSD - array (channels, time samples)
    '''
    validchs = np.where(~np.isnan(np.mean(LFPs,axis=1)))[0]
    chgroups = groupConsec(validchs,minsize=3) # to make sure only groups of at least 3 consecutive chs are used

    nChs = np.size(LFPs,0)
    nSamples = np.size(LFPs,1)

    CSD = np.zeros((nChs,nSamples))
    for chgroupi,chgroup in enumerate(chgroups):
            CSD[chgroup] = runCSD_(LFPs[chgroup])

    if smooth: CSD = smoothCSD(CSD,spacing=spacing,sm=sm)
    return CSD


def smoothCSD(csd_,spacing,sm=20):
    '''Smooths CSD.
    csd_:csd to smooth.
    spacing:spacing between channels in um
    sm:smoothing parameter
    '''

    from scipy.interpolate import interp1d

    nch = len(csd_)
    #csd_ = runCSD(trigLFP_)

    validchs = ~np.isnan(csd_[:,0])
    x = (np.arange(nch)*spacing)[validchs]

    interpolator = interp1d(x,csd_[validchs,:],kind='quadratic',axis=0)
    x_ = np.arange(x.min(),x.max()+.1)
    csd_interpl = interpolator(x_)

    csd_interpl_s = MatrixGaussianSmooth(csd_interpl.T,sm)[0].T

    interpolator = interp1d(x_,csd_interpl_s,kind='quadratic',axis=0)
    csd_s = interpolator(np.arange(nch)*spacing)

    return csd_s
# ------------------------------------------------------------------
### PLOT FUNCTIONS
def probe_yticks(layer,ax=None):
    '''Writes the corresponding layer name for each lfp trace'''
    if ax is None:
        ax = plt.gca()
    ## get layers inf
    layer_trode = []
    layer_lbls = []
    for region in list(layer):
        if np.any(layer[region]):
            tr = layer[region]
            if isinstance(layer[region],(list,np.ndarray)):
                tr = layer[region][0]
            layer_trode.append(tr)
            layer_lbls.append(region)
    ## label ###
    ax.set_yticks(layer_trode)
    ax.set_yticklabels(layer_lbls)

def get_95ci(x, ci=0.95, axis=0, n_bootstraps=5000, method='bootstraps'):
    x = np.array(x)
    # Handle cases based on the selected method
    if method == 'bootstraps':
        # Assume x is at least 2D; reshape x if it's not
        if x.ndim == 1:
            x = x.reshape(-1, 1) if axis == 0 else x.reshape(1, -1)

        # Number of elements across the specified axis
        num_elements = x.shape[axis]

        # Transpose x to bring the specified axis to the front if it's not already axis 0
        if axis != 0:
            x = np.moveaxis(x, axis, 0)

        # Initialize the results array to store CI for each element across the axis
        lower_bounds = np.empty(x.shape[1])
        upper_bounds = np.empty(x.shape[1])

        # Iterate over each slice along the axis
        bootstrap_means = np.empty((n_bootstraps, x.shape[1]))
        for i in range(n_bootstraps):
            indices = np.random.choice(num_elements, size=num_elements, replace=True)
            bootstrap_means[i, :] = np.nanmean(x[indices], axis=axis)

        lower_percentile = (1 - ci) / 2
        upper_percentile = 1 - lower_percentile
        lower_bounds = np.percentile(bootstrap_means, lower_percentile * 100, \
                                     axis=axis)
        upper_bounds = np.percentile(bootstrap_means, upper_percentile * 100, \
                                     axis=axis)

        if x.ndim == 1:
            return np.hstack([lower_bounds, upper_bounds])
        return lower_bounds, upper_bounds

    elif method == 'normal':
        lower_percentile = (1 - ci) / 2
        upper_percentile = 1 - lower_percentile
        cis = 100*np.array([lower_percentile,upper_percentile])
        # Calculate the confidence interval using the t-distribution
        ci_bounds = np.nanpercentile(x,cis,axis=axis)
        return ci_bounds

    else:
        # Calculate the mean and standard error of the mean (SEM) across the specified axis
        mean_x = np.nanmean(x, axis=axis)
        sem_x = stats.sem(x, nan_policy='omit', axis=axis)
        # Degrees of freedom adjusted for the specified axis
        df = x.shape[axis] - 1
        # Calculate the confidence interval using the t-distribution
        ci_bounds = stats.t.interval(ci, df, loc=mean_x, scale=sem_x)
        return ci_bounds
        
def plt_Mean_CI_Area(x,data,ci=0.95,axis=0,color='k',ls='-',alpha=.5,label='',ax=None,n_bootstraps=5000,method_ci='bootstraps'):
    if ax is None:
        ax = plt.gca()
    mu = np.nanmean(data,axis=axis)
    ci = get_95ci(data,ci=ci,axis=axis,n_bootstraps=n_bootstraps,method=method_ci)

    ax.plot(x,mu,ls,color=color,lw=2,label=label)
    ax.fill_between(x,ci[0],ci[1],color=color,alpha=alpha)


def AdjustBoxPlot(ax=None,alpha=0.75,color=(.6,.6,.6)):
    '''Despines axis and adds a grid'''
    import seaborn as sns
    if ax is None:
        ax = plt.gca()
    ax.grid(True,which='major',color=color,alpha=alpha,linestyle='--')
    sns.despine(ax=ax)
    
def label_plot(xlabel='',ylabel='',title='',legend=None,suptitle=None,ax=None):
    # Label axis 

    if ax is None:
        ax = plt.gca()

    ax.set_xlabel(xlabel,fontdict=fontdict)
    ax.set_ylabel(ylabel,fontdict=fontdict)
    ax.set_title(title, fontdict=fontdict)

    if legend is not None:
        ax.legend()
    if suptitle is not None:
        plt.suptitle(suptitle,fontdict=fontdict)


def plotCSD(csd, lfp, time, spacing=1, zerochi=0,\
            levels_=50, lw=1.5, xlim=None, cmap='seismic', ax_lbl=None,
            ax=None,layers=None):

    nchs = len(csd)

    if ax is None:
        ax = plt.gca()

    if xlim is None:
        xlim = [time.min(), time.max()]

    xlbl = 'Time from event (ms)'
    ylbl = 'Distance from reference layer ($\mu$m)'
    if ax_lbl is not None:
        xlbl = ax_lbl['xlabel']
        ylbl = ax_lbl['ylabel']

    tmask = (time >= np.min(xlim)) & (time <= np.max(xlim))

    if type(levels_) == int:
        climaux = np.nanmax(np.abs(csd[:, tmask]))
        levels = np.linspace(-climaux, climaux, levels_)
    else:
        levels = levels_.copy()

    csd_ = csd[:, tmask]
    taxis_ = time[tmask]

    yaxis_ = (np.arange(nchs) - zerochi) * spacing

    cnt = ax.contourf(taxis_, yaxis_, csd_, levels, cmap=cmap, extend='both')
    #for c in cnt.collections:
       # c.set_edgecolor("face")

    aux = lfp[~np.isnan(lfp)]
    scaling = 2.75 * spacing / (np.max(aux) - np.min(aux))

    for chi in range(len(lfp)):
        ax.plot(taxis_, -lfp[chi, tmask] * scaling + yaxis_[chi], 'k', linewidth=lw)

    ax.set_ylim((nchs - zerochi - 1) * spacing, -zerochi * spacing)
    ax.set_ylabel(ylbl, fontsize=16)
    ax.set_xlabel(xlbl, fontsize=16)
    ax.set_yticks(yaxis_)
    ax.set_yticklabels(yaxis_, fontsize=15)
    ax.tick_params(direction='out', labelsize=15)
    
    if layers is not None:
        probe_yticks(layers, ax=ax)