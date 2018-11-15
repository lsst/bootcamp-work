
from matplotlib import pylab as plt
import numpy

def plot_all_amps(hist_dict_list, key, **kwargs):

    xlabel = kwargs.get('xlabel', 'x')
    ylabel = kwargs.get('ylabel', 'y')
    savepng = kwargs.get('savepng')

    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    for i, ccd_hists in enumerate(hist_dict_list):
        keyed_data = ccd_hists[key]
        hist_data = keyed_data['hist']
        bin_edges = hist_data[1]
        x = (bin_edges[0:-1] + bin_edges[1:]) /2.
        y = hist_data[0]
        yerr = numpy.sqrt(y)
        model = keyed_data.get('model')
        indx = i%4
        indy = i//4
        axs[indx][indy].errorbar(x, y, yerr=yerr, c='b', linewidth=1)
        if model is not None:
            axs[indx][indy].plot(x, model, c='r', linewidth=2)
        if indx == 3:
            axs[indx][indy].set_xlabel(xlabel)
        if indy == 0:
            axs[indx][indy].set_ylabel(ylabel)

    if savepng is not None:
        fig.savefig(savepng)
    return fig, axs
