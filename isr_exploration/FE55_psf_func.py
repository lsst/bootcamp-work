import numpy as np
from scipy.special import erf

_sqrt2 = np.sqrt(2)

def pixel_integral(x, y, x0, y0, sigmax, sigmay):
    """
    Integrate 2D Gaussian centered at (x0, y0) with widths sigmax and
    sigmay over a square pixel at (x, y) with unit width.
    """
    x1, x2 = x - 0.5, x + 0.5
    y1, y2 = y - 0.5, y + 0.5

    Fx = 0.5*(erf((x2 - x0)/_sqrt2/sigmax) - erf((x1 - x0)/_sqrt2/sigmax))
    Fy = 0.5*(erf((y2 - y0)/_sqrt2/sigmay) - erf((y1 - y0)/_sqrt2/sigmay))

    return Fx*Fy

def psf_func(pos, x0, y0, sigmax, sigmay, DN_tot):
    """
    For a pixel location or list of pixel locations, pos, compute the
    DN per pixel for a 2D Gaussian with parameters:
    x0, y0: Gaussian mean x and y values
    sigmax, sigmay: Gaussian widths in x- and y-directions
    DN_tot: Gaussian normalization in ADU
    """
    return DN_tot*np.array([pixel_integral(x[0], x[1], x0, y0,
                                           sigmax, sigmay) for x in pos])


def psf_func_w_bkgd(pos, x0, y0, sigmax, sigmay, DN_tot, a, b, c):
    """
    For a pixel location or list of pixel locations, pos, compute the
    DN per pixel for a 2D Gaussian with linear background, with parameters:
    x0, y0: Gaussian mean x and y values
    sigmax, sigmay: Gaussian widths in x- and y-directions
    DN_tot: Gaussian normalization in ADU
    background = a*x + b*y + c
    """
    x1, x2 = pos[0] - 0.5, pos[0] + 0.5
    y1, y2 = pos[1] - 0.5, pos[1] + 0.5
    aI = 0.5*a*(x2*x2-x1*x1)
    bI = 0.5*b*(y2*y2-y1*y1)
    cI = c
    
    return psf_func(pos, x0, y0, sigmax, sigmay, DN_tot) + aI + bI + cI

def residuals(pars, pos, dn, errors):
    x0, y0, sigmax, sigmay, DN_tot, a, b, c = pars
    return (dn - psf_func_w_bkgd(pos, x0, y0, sigmax, sigmay, DN_tot, a, b, c))/errors

def residuals_single(pars, pos, dn, errors):
    x0, y0, sigma, DN_tot, a, b, c = pars
    return (dn - psf_func_w_bkgd(pos, x0, y0, sigma, sigma, DN_tot, a, b, c))/errors

