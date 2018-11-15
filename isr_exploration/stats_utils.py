
import numpy as np
from scipy.optimize import leastsq
#from scipy.special import erf

def single_gaussian(x, params):
    (c1, mu1, sigma1) = params
    res =  c1 * np.exp(-(x - mu1)**2.0/(2.0 * sigma1**2.0))
    return res

def single_gaussian_fit(params, x, y):
    fit = single_gaussian( x, params )
    return (fit - y)


def bin_lists(intable, bin_dict):
    o_dict = {}
    stats_dict = {}
    for key, binning in bin_dict.items():
        try:
            data = intable[key].data
        except KeyError:
            continue
        xmin, xmax, nbins = binning
        if xmin is None:
            xmin = np.min(data)
        if xmax is None:
            xmax = np.min(data)
        bins = np.linspace(xmin, xmax, nbins+1)
        hist = np.histogram(data, bins=bins)
        if len(data) > 0:
            mean = np.mean(data)
            stdev = np.std(data)
        else:
            mean = 0.
            stdev = 0.
            hist = None
        o_dict[key] = dict(hist=hist,
                           mean=mean,
                           stdev=stdev)
    return o_dict


def fit_hists_gaussian(hist_dict, keys):

    for key in keys:
        data = hist_dict[key]
        hist = data['hist']
        if hist is None:
            fit = None
            model = None
        else:
            mean = data['mean']
            stdev = data['stdev']
            y = hist[0]
            x = [(hist[1][i]+hist[1][i+1])/2 for i in range(len(hist[1])-1)]
            norm = np.sum(y)
            pars = (norm, mean, stdev)
            fit = leastsq(single_gaussian_fit, pars, args=(x, y))
            model = single_gaussian(x, fit[0])
        hist_dict[key]['fit'] = fit
        hist_dict[key]['model'] = model
