
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


def bin_lists(input_dict, bin_dict):
    o_dict = {}
    stats_dict = {}
    for key, val_list in input_dict.items():
        try:
            xmin, xmax, nbins = bin_dict[key]
        except KeyError:
            continue
        bins = np.linspace(xmin, xmax, nbins+1)
        hist = np.histogram(val_list, bins=bins)
        if len(val_list) > 0:
            mean = np.mean(val_list)
            stdev = np.std(val_list)        
            y = hist[0]
            x = [(hist[1][i]+hist[1][i+1])/2 for i in range(len(hist[1])-1)]
            norm = np.sum(y)
            pars = (norm, mean, stdev)
            fit = leastsq(single_gaussian_fit, pars, args=(x, y))
        else:
            mean = 0.
            stdev = 0.
            fit = None
        o_dict[key] = dict(hist=hist,
                           mean=mean,
                           stdev=stdev,
                           fit=fit)
    return o_dict

