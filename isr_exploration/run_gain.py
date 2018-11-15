from multiprocessing import Pool
# system imports
from matplotlib import pylab as plt
import numpy as np
import os
import sys
import pickle

# LSST stack imports
from lsst.daf.persistence import Butler
import lsst.afw.display as afwDisplay
from lsst.ip.isr import IsrTask
import lsst.afw.math as afwMath

import eups
assert eups.getSetupVersion("obs_lsst")

import warnings
warnings.filterwarnings("ignore")

BOOTCAMP_REPO_DIR= '/project/bootcamp/repo_RTM-011/'

def single_gains(detector_num):
    
    isr_config = IsrTask.ConfigClass()

    isr_config.doDark=False
    isr_config.doFlat=False
    isr_config.doFringe=False
    isr_config.doDefect=False
    isr_config.doAddDistortionModel=False
    isr_config.doLinearize=False
    isr_config.doSaturationInterpolation=False

    isr = IsrTask(config=isr_config)
    
    
    butler = Butler(BOOTCAMP_REPO_DIR)
    visits = butler.queryMetadata('raw', ['visit'], dataId={'imageType': 'FLAT', 'testType': 'FLAT'})
    amp_info_007 = {}
    gain = {}
    exp_time = {}
    mean_brightness = {}

    amp_info_007 = {}
    gain = {}
    exp_time = {}
    mean_brightness = {}
    
    bias = None
    
    i = 1
    #visits = visits[:8]
    
    for visit in visits: # loop over pairs of images
        # Get ISR data for first image
        dId = {'visit': visit, 'detector': int(detector_num)}
        raw = butler.get('raw', dId)
        if (bias == None):
            bias = butler.get('bias', dId)

        # run ISR on both images
        result = isr.run(raw, bias=bias)

        detector = result.exposure.getDetector()

        if i == 1:
            for amp_number in range(len(detector)):
                amp_info_007[amp_number] = {'gain':{}, 'exp_time':{}, 'mean_brightness':{}}

        for amp_number in range(len(detector)):
            amp = detector[amp_number]

            sub_im1 = result.exposure.getMaskedImage()[amp.getBBox()]
            #arr1 = sub_im1.getImage().getArray()
            sub_im2 = np.full(sub_im1.shape(), np.mean(sub_im1))
            #arr2 = sub_im2.getImage().getArray()

            # From RHL, 1/g = <(I1-I2)**2/(I1+I2)>
            diff_im = sub_im1.clone()
            diff_im -= sub_im2

            sum_im = sub_im1.clone()
            sum_im += sub_im2

            diff_im *= diff_im
            diff_im /= sum_im

            stats = afwMath.makeStatistics(diff_im, afwMath.MEDIAN | afwMath.MEAN)
            # Compute gain for this amp.


            amp_info_007[amp_number]['gain'][visit1] = 1/stats.getValue(afwMath.MEAN)
            amp_info_007[amp_number]['exp_time'][visit1] = time1
            stats_sum = afwMath.makeStatistics(sum_im, afwMath.MEAN)
            amp_info_007[amp_number]['mean_brightness'][visit1] = stats_sum.getValue(afwMath.MEAN)/2.
            #print("visit %i,%i -- amp %i -- %i of %i -- gain=%f, exposure time(s)=%f"%(visit1, visit2, amp_number, i, len(visits)/2, 
            #                                                                           amp_info_007[amp_number]['gain'][visit1], 
            #                                                                           amp_info_007[amp_number]['exp_time'][visit1]))

        print("amp %i -- %i of %i -- detector %i"%(amp_number, i, len(visits)/2, detector_num))
        i += 1
        
    return (detector_num, amp_info_007)


def pair_gains(detector_num):

    isr_config = IsrTask.ConfigClass()

    isr_config.doDark=False
    isr_config.doFlat=False
    isr_config.doFringe=False
    isr_config.doDefect=False
    isr_config.doAddDistortionModel=False
    isr_config.doLinearize=False
    isr_config.doSaturationInterpolation=False

    isr = IsrTask(config=isr_config)
    
    butler = Butler(BOOTCAMP_REPO_DIR)
    visits = butler.queryMetadata('raw', ['visit'], dataId={'imageType': 'FLAT', 'testType': 'FLAT'})
    amp_info_007 = {}
    gain = {}
    exp_time = {}
    mean_brightness = {}

    amp_info_007 = {}
    gain = {}
    exp_time = {}
    mean_brightness = {}
    
    bias = None
    
    i = 1
    #visits = visits[:8]
    
    for visit1, visit2 in zip(visits[:-1:2], visits[1::2]): # loop over pairs of images
        # Get ISR data for first image
        dId = {'visit': visit1, 'detector': int(detector_num)}
        raw1 = butler.get('raw', dId)
        if (bias == None):
            bias = butler.get('bias', dId)
        time1 = raw1.getInfo().getVisitInfo().getExposureTime()

        # Get ISR data for second image
        dId = {'visit': visit2, 'detector': int(detector_num)}
        raw2 = butler.get('raw', dId)
        time2 = raw2.getInfo().getVisitInfo().getExposureTime()
        if abs(time1 - time2) > 0.01:
            "Mismatched exptimes"
            continue

        # run ISR on both images
        result1 = isr.run(raw1, bias=bias)
        result2 = isr.run(raw2, bias=bias)

        detector = result1.exposure.getDetector()

        if i == 1:
            for amp_number in range(len(detector)):
                amp_info_007[amp_number] = {'gain':{}, 'exp_time':{}, 'mean_brightness':{}}

        for amp_number in range(len(detector)):
            amp = detector[amp_number]

            sub_im1 = result1.exposure.getMaskedImage()[amp.getBBox()]
            #arr1 = sub_im1.getImage().getArray()
            sub_im2 = result2.exposure.getMaskedImage()[amp.getBBox()]
            #arr2 = sub_im2.getImage().getArray()

            # From RHL, 1/g = <(I1-I2)**2/(I1+I2)>
            diff_im = sub_im1.clone()
            diff_im -= sub_im2

            sum_im = sub_im1.clone()
            sum_im += sub_im2

            diff_im *= diff_im
            diff_im /= sum_im

            stats = afwMath.makeStatistics(diff_im, afwMath.MEDIAN | afwMath.MEAN)
            # Compute gain for this amp.


            amp_info_007[amp_number]['gain'][visit1] = 1/stats.getValue(afwMath.MEAN)
            amp_info_007[amp_number]['exp_time'][visit1] = time1
            stats_sum = afwMath.makeStatistics(sum_im, afwMath.MEAN)
            amp_info_007[amp_number]['mean_brightness'][visit1] = stats_sum.getValue(afwMath.MEAN)/2.
            #print("visit %i,%i -- amp %i -- %i of %i -- gain=%f, exposure time(s)=%f"%(visit1, visit2, amp_number, i, len(visits)/2, 
            #                                                                           amp_info_007[amp_number]['gain'][visit1], 
            #                                                                           amp_info_007[amp_number]['exp_time'][visit1]))

        print("amp %i -- %i of %i -- detector %i"%(amp_number, i, len(visits)/2, detector_num))
        i += 1
        
    return (detector_num, amp_info_007)

if __name__ == "__main__":

    func = pair_gains
    output = "det_data.pkl"
    if "--single" in sys.argv:
        func = single_gains
    if "-f" in sys.argv:
        output = sys.argv[sys.argv.index("-f") + 1]
    if "-repo" in sys.argv:
        BOOTCAMP_REPO_DIR= sys.argv[sys.argv.index("-repo") + 1]
    
    detector_dict_007 = {}

    pool = Pool(processes=2)              # start 2 worker processes

    results = pool.map_async(func, range(9))
    get_results = results.get()

    results_list = []
    for result_on in get_results:
        detector_dict_007[result_on[0]] = result_on[1]
        
    pickle.dump(detector_dict_007, open(output, "wb" ))