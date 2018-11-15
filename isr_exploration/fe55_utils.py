
import numpy as np
from scipy.optimize import leastsq
import FE55_psf_func
from scipy.special import erf, gammaincc

def fit_single_footprint(fp, **kwargs):

    # input arguments
    npars = kwargs.get('npars', 5)
    min_npix = kwargs.get('min_npix', 9)
    max_npix = kwargs.get('max_npix', 100)
    stdev = kwargs.get('stdev')
    sigma0 = kwargs.get('sigma0', 0.36)
    dn0 = kwargs.get('dn0', 1590./5.)
    
    # input data
    imarr = kwargs.get('imarr')
    
    # variables we are filling
    zvals = kwargs.get('zvals', [])
    sigmax = kwargs.get('sigmax', [])
    sigmay = kwargs.get('sigmay', [])
    dn = kwargs.get('dn', [])
    x0 = kwargs.get('x0', [])
    y0 = kwargs.get('y0', [])
    aList = kwargs.get('a', [])
    bList = kwargs.get('b', [])
    cList = kwargs.get('c', [])
    dn_fp = kwargs.get('dn_fp', [])
    chiprob = kwargs.get('chiprob', [])
    chi2 = kwargs.get('chi2', [])
    dof = kwargs.get('dof', [])
    maxDN = kwargs.get('maxDN', [])
    p9_data = kwargs.get('p9_data', [])
    p9_model = kwargs.get('p9_model', [])
    prect_data = kwargs.get('prect_data', [])
    xpeak = kwargs.get('xpeak', [])
    ypeak = kwargs.get('ypeak', [])
    
    if fp.getArea() < min_npix or fp.getArea() > max_npix:            
        return -1
    spans = fp.getSpans()
    positions = []
    zvals = []
    peak = [pk for pk in fp.getPeaks()][0]
    dn_sum = 0
    a_0 = 0.0
    b_0 = 0.0
    c_0 = 100.
    
    for span in spans:
        y = span.getY()
        for x in range(span.getX0(), span.getX1() + 1):
            ym = y%imarr.shape[0]
            xm = x%imarr.shape[1]
            zvals.append(imarr[ym][xm])
            dn_sum += imarr[ym][xm]
            positions.append((x, y))
                
    try:
        # Use clipped stdev as DN error estimate for all pixels
        dn_errors = stdev*np.ones(len(positions))
        if npars == 5:
            p0 = (peak.getIx(), peak.getIy(), sigma0, sigma0, dn0, a_0, b_0, c_0)
            pars, _ = leastsq(FE55_psf_func.residuals, p0, args=(positions, zvals, dn_errors))
            sigx = pars[2]
            sigy = pars[3]
            dnval = pars[4]
            aval = pars[5]
            bval = pars[6]
            cval = pars[7]
        else:
            p0 = (peak.getIx(), peak.getIy(), sigma0, dn0, a_0, b_0, c_0)
            pars, _ = leastsq(FE55_psf_func.residuals_single, p0, args=(positions, zvals, dn_errors))
            sigx = pars[2]
            sigy = pars[2]
            dnval = pars[3]
            aval = pars[4]
            bval = pars[5]
            cval = pars[6]
        chi2val = FE55_psf_func.chisq(positions, zvals, pars[0], pars[1], sigx, sigy, dnval, aval, bval, cval, dn_errors)
        dofval = fp.getArea() - npars
        prob = gammaincc(dofval/2., chi2val/2.)        
        if prob < 1e-2:
            return -1
        if npars == 5:            
            sigmax.append(pars[2])
            sigmay.append(pars[3])
            dn.append(pars[4])
            aList.append(pars[5])
            bList.append(pars[6])
            cList.append(pars[7])
        else:
            sigmax.append(pars[2])
            sigmay.append(pars[2])
            dn.append(pars[3])
            aList.append(pars[4])
            bList.append(pars[5])
            cList.append(pars[6])
        x0.append(pars[0])
        y0.append(pars[1])
        dn_fp.append(dn_sum)
        chi2.append(chi2val)
        dof.append(dofval)
        chiprob.append(prob)
        maxDN.append(max(zvals))
        try:
            #p9_data_row, p9_model_row \
            #    = p9_values(peak, imarr, x0[-1], y0[-1], sigmax[-1],
            #                sigmay[-1], dn[-1])
            #prect_data_row = prect_values(peak,imarr)
            #p9_data.append(p9_data_row)
            #p9_model.append(p9_model_row)
            #prect_data.append(prect_data_row)
            xpeak.append(peak.getIx())
            ypeak.append(peak.getIy())
        except IndexError:
            [item.pop() for item in (x0, y0, sigmax, sigmay,
                                    dn, aList, bList, cList, dn_fp, chiprob,
                                    chi2s, dofs, maxDNs)] 
    except RuntimeError:
        return -1
            
    return 0


