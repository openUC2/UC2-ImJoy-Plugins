#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 11:24:58 2021

@author: bene
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import numbers
from scipy.ndimage import gaussian_filter

def GLS(x,y,v):
    """
    [o,s]=GLS(x,y,v)  : Generalized Least Square fit, linear regression with error. Fits a straight line with offset through data with known variances.
    x : vector of x-values, known postions at which was measured.
    y : vector y-values, measurements.
    v : vector of known errors. These can also be estimated, but if you have a model, use RWLS, the reweighted least squares regression.
        Specifically look at RWLSPoisson if this fitting is for Poisson statistics.
    o : offset = y-value at zero x-value
    s : slope

    This routine is based on the Wikipedia description at
    https://en.wikipedia.org/wiki/Linear_regression
    (Xt Omega-1 X)^-1  Xt Omega^-1  Y
    Example:
    [o,s]=GLS([1 2 3 4],[7 8 9 11],[1 1 1 1])
    """
    Xt = np.stack((np.ones(x.shape),x),-2)
    #X = np.transpose(Xt)
    Omega_X = np.transpose(np.stack((1/v,x/v),-2))
    Omega_Y = np.transpose(y/v)
    ToInvert = Xt.dot(Omega_X) # matrix product

    res = np.linalg.inv(ToInvert).dot(Xt.dot(Omega_Y))
    return res


def RWLSPoisson(x,y,N=1, ignoreNan=True, validRange=None):
    """ [o,s]=RWLSPoisson(x,y,N)  : Poisson Reweighted Least Square fit, linear regression with error according to the Poisson distribution. Fits a straight line with offset.
     x : vector of x-values, known postions at which was measured.
     y : vector y-values, measurements.
     N : optional number of measurements from which y was obtained by averaging
     ignoreNan : should Nan valued be irnored?
     validRange : optionally limits the range to fit

     This routine is based on the Wikipedia description at
     https://en.wikipedia.org/wiki/Linear_regression
     (Xt Omega-1 X)^-1  Xt Omega^-1  Y

     Example:
     [o,s]=RWLSPoisson([1 2 3 4],[7 8 9 11])
    """
    if ignoreNan:
        valid = ~ np.isnan(y)
        x = x[valid]
        y = y[valid]
        if len(np.shape(N)) > 0:
            N = N[valid]

    if validRange is not None:
        allvv = x * np.nan
        valid = (x <= validRange[1]) & (x >= validRange[0])
        x = x[valid]
        y = y[valid]
        if len(np.shape(N)) > 0:
            N = N[valid]

    myThresh = 1.0 # roughly determined by a simulation
    NumIter = 5
    v = y  # variances of data is equal (or proportional) to the measured variances
    for n in range(NumIter):
        vv = v**2 / N; # Variance of the variance. The error of the variance is proportional to the square of the variance, see http://math.stackexchange.com/questions/1015215/standard-error-of-sample-variance
        if any(v<myThresh):
            vv[v<myThresh]=myThresh;  # This is to protect agains ADU-caused bias, which is NOT reduced by averaging
            if n == 1:
                print('WARNING RWLSPoisson: The data has a variance below 2 ADUs at low signal level. This leads to unwanted biases. Increasing the variance estimation for the fit.\n');
        (o,s) = GLS(x, y, vv)
        v = o+s*x # predict variances from the fit for the next round
        # print('RWLSPoisson Iteration %d, o: %g, s: %g\n',n,o,s);
    if validRange is not None:
        allvv[valid] = vv
        vv = allvv
    return o, s, vv


def cal_readnoise(fg,bg,numBins=100, validRange=None, CameraName=None, correctBrightness=True,
 correctOffsetDrift=True, excludeHotPixels=True, crazyPixelPercentile=98, doPlot=True, exportpath=None,
 brightness_blurring=True, plotWithBgOffset=True):
    """
    calibrates detectors by fitting a straight line through a mean-variance plot
    :param fg: A series of foreground images of the same (blurry) scene. Ideally spanning the useful range of the detector. Suggested number: 20 images
    :param bg: A series of dark images under identical camera settings (integration time, temperature, etc.). Suggested number: 20 images
    :param numBins: Number of bins for the fit
    :param validRange: If provided, the fit will only be performed over this range (from,to) of mean values. The values are inclusived borders
    :param doPlot: Plot the mean-variance curves
    :return: tuple of fit results (offset [adu], gain [electrons / adu], readnoise [e- RMS])
    to scale your images into photons use: (image-offset) * gain
    """
    function_args = locals()

    figures = [] # collect figures for returning
    # define plotting parameters
    CM = 1/2.54 # centimetres for figure sizes
    plot_figsize = 26*CM, 15*CM # works well for screens
    AxisFontSize=12
    TextFontSize=8
    TitleFontSize=14
    # rc('font', size=AxisFontSize)  # controls default text sizes
    # rc('axes', titlesize=AxisFontSize, labelsize=AxisFontSize)  # fontsize of the axes title and number labels
    
    
    fg = np.squeeze(fg).astype(float)
    bg = np.squeeze(bg).astype(float)
    didReduce = False
    if fg.ndim > 3:
        print('WARNING: Foreground data has more than 3 dimensions. Just taking the first of the series.')
        fg = fg[0]
        didReduce = True
    if bg.ndim > 3:
        print('WARNING: Background data has more than 3 dimensions. Just taking the first of the series.')
        bg = bg[0]
        didReduce = True
    fg = np.squeeze(fg)
    bg = np.squeeze(bg)
    if didReduce:
        # return cal_readnoise(fg,bg,numBins, validRange, CameraName, correctBrightness, correctOffsetDrift, excludeHotPixels, doPlot)
        # TODO: Needs testing
        return cal_readnoise(fg,bg, *list(function_args.values()))


    Text = "Analysed {nb:d} bright and {nd:d} dark images:\n".format(nb=fg.shape[0],nd=bg.shape[0])

    validmap = np.ones(bg.shape[-2:], dtype=bool)
    # validmap = None
    if validRange is None:
        underflow = np.sum(fg <= 0, (0,)) > 0
        validmap *= ~underflow
        numUnderflow = np.sum(underflow)
        if np.min(fg) <= 0:
            print("WARNING: "+str(numUnderflow)+" pixels with at least one zero-value (value<=0) were detected but no fit range was selected. Excluded those from the fit.")
        Text = Text + "Zero-value pixels: {zv:d} excluded.\n".format(zv=numUnderflow)
        
        overflow = 0
        relsat = 0
        maxvalstr = ""
        for MaxVal in [255,1023,4095,65535]:
            if np.max(fg) == MaxVal:
                overflow = np.sum(fg == MaxVal,(0,)) > 0
                relsat = np.sum(overflow)
                print("WARNING: "+str(relsat)+" pixels saturating at least once (value=="+str(MaxVal)+") were detected but no fit range was selected. Excluding those from the fit.")
                validmap = validmap & ~ overflow
                maxvalstr = "(=="+str(MaxVal)+")"
        Text = Text + "Overflow "+ maxvalstr +" pixels: "+str(relsat)+"  excluded.\n"

    print("Calibration results:")
    if correctOffsetDrift:
        meanoffset = np.mean(bg, (-2,-1), keepdims=True)
        refOffset = np.mean(bg)
        bg = bg - meanoffset + refOffset
        reloffset = (refOffset - meanoffset)  / np.sqrt(np.var(bg))
        if doPlot:
            fig = plt.figure(figsize=plot_figsize)
            if CameraName is not None:
                plt.title("Offset Drift ("+CameraName+")", fontsize=TitleFontSize)
            else:
                plt.title("Offset Drift", fontsize=TitleFontSize)
            plt.plot(reloffset.flat, label='Offset / Std.Dev.')
            plt.xlabel("frame no.", fontsize=AxisFontSize)
            plt.ylabel("mean offset / Std.Dev.", fontsize=AxisFontSize)

            figures.append(fig)
            if exportpath is not None:
                plt.savefig(exportpath/'correctOffsetDrift.png')


    bg_mean_projection = np.mean(bg, (-3))
    bg_total_mean = float(np.mean(bg_mean_projection)) # don't want to return image type
    plotOffset = bg_total_mean*plotWithBgOffset
    patternVar = np.var(bg_mean_projection)

    fg -= bg_mean_projection # pixel-wise background subtraction
    if correctBrightness:
        brightness = np.mean(fg, (-2,-1), keepdims=True)
        meanbright = np.mean(brightness)
        relbright = brightness/meanbright
        if doPlot:
            fig = plt.figure(figsize=plot_figsize)
            if CameraName is not None:
                plt.title("Brightness Fluctuation ("+CameraName+")", fontsize=TitleFontSize)
            else:
                plt.title("Brightness Fluctuation", fontsize=TitleFontSize)
            plt.plot(relbright.flat)
            plt.xlabel("frame no.",fontsize=AxisFontSize)
            plt.ylabel("relative brightness",fontsize=AxisFontSize)
            figures.append(fig)            
            if exportpath is not None:
                plt.savefig(exportpath/'Brightness_Fluctuation.png')
        fg = fg / relbright
        maxFluc = np.max(np.abs(1.0-relbright))
        Text = Text + "Illumination fluctuation: {bf:.2f}".format(bf=maxFluc * 100.0)+"%\n"
    
    fg_mean_projection = np.mean(fg, (-3)) 
    fg_var_projection = np.var(fg, (-3))
    bg_var_projection = np.var(bg, (-3))

    hotPixels = None
    if excludeHotPixels:
        hotPixels = np.abs(bg_mean_projection - np.mean(bg_mean_projection)) > 4.0*np.sqrt(np.mean(bg_var_projection))
        numHotPixels = np.sum(hotPixels)
        Text = Text + "Hot pixels (|bg mean| > 4 StdDev): "+str(numHotPixels)+" excluded.\n"
        validmap *= ~hotPixels


    noisyPixelThreshold = np.percentile(bg_var_projection, crazyPixelPercentile)
    noisyPixels = bg_var_projection > noisyPixelThreshold # need to exclude hot pixels, which skew variance
    Text = Text + "{:.0f}% of bg Pixels have a var > {:.0f} and were excluded\n".format(100-crazyPixelPercentile, noisyPixelThreshold)
    validmap *= ~noisyPixels

    # if validRange is given, it is for for biased image
    # we need to correct it for the unbiased image
    if validRange is not None: 
        validRange = np.array(validRange)
        validRange = validRange - bg_total_mean

    if brightness_blurring:
        # sCMOS brightnesses fluctuate too much, we need a filter
        # blur image, yielding better estimate for local brightness
        blurred = fg_mean_projection
        # Generate median projection to use to fill gaps from invalid pixels 2021-04 
        median_projection = scipy.ndimage.median_filter(fg_mean_projection, size=(7,7))
        blurred[~validmap] = median_projection[~validmap]
        #        blurred = gaussf(blurred, (7,7))
        blurred =  gaussian_filter(blurred, sigma=1)
        fg_mean_projection = blurred[validmap] # Note that this also excludes the invalid pixels from the plot
    else:
        fg_mean_projection = fg_mean_projection[validmap]

    fg_var_projection = fg_var_projection[validmap]
    # bg_var_projection = bg_var_projection[validmap] # this is unnecessary and would falsify the readnoise estimate

    if validmap is not None:
        # create histRange, otherwise numpy.histogram will allocate bins right up to the hot pixels
        # validMeans = fg_mean_projection[validmap] # now that it is applied above, we don't need to use validmap here
        validMeans = fg_mean_projection
        histRange = np.min(validMeans), np.max(validMeans)

        # Automatic range feature. Validrange between 99th percentile and 5% of 99th percentile
        if validRange is None:
            validRange = np.empty(2)
            validRange[1] = np.percentile(validMeans, 99)
            validRange[0] = 0.05*(validRange[1]-histRange[0])

        # correct the numBins for the validRange??. Not happy with this inconsistency @
        numBins = int(numBins*np.ptp(histRange)/np.ptp(validRange))


    else:
        histRange = None

    # binning
    # hist_num: total number of pixels in the bin
    # hist_mean_sum: sum of all the mean values of pixels in the bin
    # hist_var_sum: sum of all the variance values of pixels in the bin
    (hist_num, mybins) = np.histogram(fg_mean_projection, range=histRange, bins=numBins)
    (hist_mean_sum, mybins) = np.histogram(fg_mean_projection, range=histRange, bins=numBins, weights=fg_mean_projection)
    (hist_var_sum, mybins) = np.histogram(fg_mean_projection, range=histRange, bins=numBins, weights=fg_var_projection)

    binMid = (mybins[1:] + mybins[:-1]) / 2.0

    valid = hist_num > 0
    hist_var_sum = hist_var_sum[valid]
    hist_mean_sum = hist_mean_sum[valid]
    hist_num = hist_num[valid]
    binMid = binMid[valid]

    mean_var = hist_var_sum / hist_num # mean of variances within the bin
    mean_mean = hist_mean_sum / hist_num # mean of means within the bin

    (offset, slope, vv) = RWLSPoisson(mean_mean, mean_var, hist_num, validRange=validRange)

    myFit = binMid * slope + offset
    myStd = np.sqrt(vv)
    gain = float(1.0 / slope) # don't want to return image type
    mean_el_per_exposure = np.sum(fg.mean((-3)))*gain

    Text = Text + "Background [ADU]: {bg:.2f}".format(bg=bg_total_mean) + "\n"
    Text = Text + "Gain [e- / ADU]): {g:.4f}".format(g=gain) + "\n"
    if offset < 0.0:
        Text = Text + "Readnoise (fit): variance ({of:.2f}) below zero.".format(of=offset) + "\n"
    else:
        Text = Text + "Readnoise (fit): {rn:.2f}".format(rn=np.sqrt(offset)*gain)+"\n"
    Text = Text + "Fixed pattern offset (gain * std. dev. for mean_bg): {rn:.2f} e- RMS\n".format(rn=np.sqrt(patternVar))
    Readnoise = (np.sqrt(np.mean(bg_var_projection)) * gain).astype(float) # don't want to return image type
    Text = Text + "Readnoise, gain * bg_noise: {rn:.2f} e- RMS\n".format(rn=Readnoise)
    # median_readnoise = (np.sqrt(np.median(bg_var_projection)) * gain).astype(float) # don't want to return image type
    # Text = Text + "Median readnoise, gain * bg_noise: {rn:.2f} e- median\n".format(rn=median_readnoise)
    Text = Text + "Total electrons per exposure: {:.3E} e- \n".format(mean_el_per_exposure)

    if doPlot:
        fig = plt.figure(figsize=plot_figsize)

        if CameraName is not None:
            plt.title("Photon transfer curve ("+CameraName+")", fontsize=TitleFontSize)
        else:
            plt.title("Photon transfer curve", fontsize=TitleFontSize)

        # plotX = binMid + plotOffset
        biased_binMid = binMid + plotOffset
        biased_mean_mean = mean_mean + plotOffset
        plt.plot(biased_mean_mean, mean_var, 'bo', label='Brightness bins')
        # plt.errorbar(biased_binMid, myFit, myStd, label='Fit')
        plt.plot(biased_binMid, myFit, 'r', label='Linear fit')
        plt.plot(biased_binMid, myFit+myStd/2, '--r', label="Error")
        plt.plot(biased_binMid, myFit-myStd/2, '--r')
        plt.legend()

        # secondary axes in photoelectrons
        ax = plt.gca()
        def adu2el(adu):
            return (adu-bg_total_mean)*gain
        def el2adu(el):
            return el/gain+bg_total_mean
        secax_x = ax.secondary_xaxis('top', functions=(adu2el, el2adu))
        secax_y = ax.secondary_yaxis('right', functions=(lambda x: x*gain**2, lambda x: x/gain**2))
        
        secax_x.set_xlabel("Pixel brightness / $photoelectrons$", fontsize=AxisFontSize)
        secax_y.set_ylabel("Pixel variance / $photoelectrons^2$", fontsize=AxisFontSize)

        plt.xlabel("Pixel brightness / $ADU$", fontsize=AxisFontSize)
        plt.ylabel("Pixel variance / $ADU^2$", fontsize=AxisFontSize)
        plt.grid()
        plt.figtext(0.02, 0.05, Text, fontsize=TextFontSize)
        fig.subplots_adjust(left=0.4)
        figures.append(fig)
        if exportpath is not None:
            plt.savefig(exportpath/'Photon_Calibration.png')

        # import pdb
        # pdb.set_trace()
    if exportpath is not None:
        with open(exportpath/'calibration_results.txt', "w") as outfile:
            outfile.write(Text)
    print(Text)
    return (bg_total_mean, gain, Readnoise, mean_el_per_exposure, validmap, figures, Text)
