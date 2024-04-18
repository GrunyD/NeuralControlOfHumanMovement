"""
This script follows article 
    Feature Analysis for Classification of Physical Actions Using Surface EMG Data
    Anish C. Turlapaty and Balakrishna Gokaraju
    IEEE sensors journal
    https://ieeexplore.ieee.org/document/8817944

"""

import numpy as np
from scipy import stats, signal

LBP_THRESHOLD = 500
LBP_STRIDE = 10
LBP_WINDOW = None #If none the sliding window will take size of the whole window

SAMPLING_FREQUENCY = 25

NBANDS = 10

def timeDomainStats(emgData: np.ndarray) -> np.ndarray:
    """
    Parameters:
    ------------
    - emgData: shape (channels, samples)

    Returns:
    ---------
    - timeDomainStatistics: shape (channels * nFeatures, )
    """

    axis = 1
    mean = np.average(emgData, axis = axis, keepdims=True)
    variance = np.std(emgData, axis = axis, keepdims=True)**2
    skewness = stats.skew(emgData, axis = axis, keepdims=True)
    kurtosis = stats.kurtosis(emgData, axis = axis, keepdims=True)

    timeDomainStatistics = np.hstack((mean, variance, skewness, kurtosis))
    return timeDomainStatistics.flatten()

def interchannelStats(emgData: np.ndarray, mode:str = "full") -> np.ndarray:
    """
    Calculates correlation between all unique pairs 
    of channels and takes the maximal value for each

    Parameters:
    ------------
    - emgData: shape (channels, samples)
    - mode: is passed to scipy.signal.correlation function, 
        possible values are 'full', 'valid', 'same'

    Returns:
    ---------
    - interchannelStats: shape (nUniquePairsOfChannels,) 
    """
    indices = np.arange(stop=emgData.shape[0])

    values = []
    for index1 in indices[:-1]:
        for index2 in indices[index1+1:]:
            correlation = signal.correlate(emgData[index1,:], emgData[index2, :], mode = mode)
            values.append(np.max(correlation))
    return np.array(values)


def logMomentsFourierSpectra(emgData: np.ndarray) -> np.ndarray:
    """
    Parameters:
    ------------
    -emgData: shape (channels, samples)

    Returns:
    ------------
    - features: shape (17*channels, )
    """

    spectra = np.fft.fft(emgData, axis = 1)
    powerSpectra = np.absolute(spectra)**2 #equation 8
    k = np.arange(1, powerSpectra.shape[1]+1)
    k = np.broadcast_to(k, powerSpectra.shape)
    # print(k.shape)
    # print((k<0).any())
    def domainMoment(i):
        inside = np.sum(powerSpectra * k**i, axis = 1)
        # if (inside < 0).any():
        #     print(inside)
        return np.sqrt(inside)
    # domainMoment = lambda i: np.sqrt(np.sum(powerSpectra * k**i, axis = 1)) #equation 9

    g0 = domainMoment(0)
    g2 = domainMoment(2)
    features = [
        np.log(g0),
        np.log(g2),
        np.log(domainMoment(4)),
        np.log(g0) - 0.5*np.log(g0 - g2) - 0.5 * np.log(g0 - domainMoment(4)),
        np.log(g2) - 0.5*np.log(g0*domainMoment(4)),
        np.log(g0) - 0.25*np.log(domainMoment(1)*domainMoment(3)),
        np.log(g0) - 0.25*np.log(g2*domainMoment(6))
    ] #equation 10

    pairs = [
        (1,2), (1,3), (1,4), (1,5),
        (2,3), (2,4), (2,5),
        (3,4), (3,5),
        (4,5)
    ] #equation 12

    feature = lambda n: 0.5 * np.log(domainMoment(pairs[n-8][0]) * domainMoment(pairs[n-8][1])) 
    #equation 11
    # This feature calculation is for n >= 8, for n < 8 features are defined in the list 'features'
    # Based on n it takes corresponding pair of indexes and computes the domain moment of this index

    for n in range(8, 18):
        features.append(feature(n))

    return np.array(features).flatten()

def spectralBandPowers(emgData:np.ndarray, samplingFrequency:int = SAMPLING_FREQUENCY, nBands:int = NBANDS) -> np.ndarray:
    """
    Not sure about the whole thing, was not the best description in the paper

    Parameters:
    ------------
    - emgData: shape (channels, samples)
    - samplingFrequency
    - nBands: number of features per channel

    Returns:
    -----------
    - features: shape(channels*bands, )

    TODO: improve how we decide on the bands
    """
    f, powerSpectrum = signal.welch(emgData, fs = samplingFrequency, axis = 1, nperseg = emgData.shape[1])
    bands = np.array_split(powerSpectrum, nBands, axis = 1,)
    features = np.empty((emgData.shape[0], nBands))
    for index, band in enumerate(bands):
        features[:,index] = np.sum(band, axis = 1)
    return features.flatten()



def localBinaryPatterns(emgData:np.ndarray, slidingWindowLength:int = LBP_WINDOW, stride:int = LBP_STRIDE, threshold:int = LBP_THRESHOLD) -> np.ndarray:
    """
    Parameters:
    ------------
    - emgData: shape (channels, samples)
    - slidingWindowLenght: How big is the sliding window for local binary patterns
        if None it will make just one big window
    - stride: stride of the sliding window
    - threshold: final thresholding as described in th epaper

    Returns:
    -----------
    features: shape (channels * 2, )
    """
    
    if slidingWindowLength is None:
        slidingWindowLength = emgData.shape[1]

    LBPs = []
    position = 0
    while slidingWindowLength + position <= emgData.shape[1]:
        subset = emgData[:, position:slidingWindowLength + position]
        mean = np.average(subset, axis = 1, keepdims=True)
        LBPs.append(np.sum(subset**(subset >= mean), axis = 1)) #equation 16
        position += stride
    LBPs = np.array(LBPs).T
    features1 = np.sum(LBPs > threshold, axis = 1)
    features2 = np.sum(LBPs <= threshold, axis = 1)

    return np.concatenate((features1, features2))



def getFeatures(emgData: np.ndarray, featureCalculationFuncs:list = None, segmentWindowLength:int = None) -> np.ndarray:
    """
    Parameters:
    ------------
    - emgData: shape (channels, samples)
    - featureCalculationFuncs: List caontaining functions that calculate desired features,
        Functions should only take in one argument 'emgData'
    - segmentWindowLength: The paper describes calculating the features of signal repeatedly 
        over non overlapping time window. If null features are calculated for the whole signal

    Returns:
    ---------
    - features: 1D array containing all the calculated features

    """

    def getSegmentFeatures(emgData: np.ndarray, funcs:list):
        features = []
        for func in funcs:
            features.append(func(emgData))
        return np.concatenate(features)
    
    if featureCalculationFuncs is None:
        featureCalculationFuncs = [
            # timeDomainStats,
            interchannelStats, # No error
            # logMomentsFourierSpectra,
            # localBinaryPatterns, # No error
            # spectralBandPowers # No error
        ]
        
    if segmentWindowLength is None or segmentWindowLength > emgData.shape[1]:
        return getSegmentFeatures(emgData, featureCalculationFuncs)
    
    r = emgData.shape[1] // segmentWindowLength
    indices = np.array([i * segmentWindowLength for i in range(1, r)])
    emgSegments = np.split(emgData, indices_or_sections= indices, axis = 1)

    segmentsFeatures = []
    for emgSegment in emgSegments:
        segmentsFeatures.append(getSegmentFeatures(emgSegment, featureCalculationFuncs))
 

    return np.concatenate(segmentsFeatures)

    

     

if __name__ == "__main__":
    i = np.arange(50)
    i = np.broadcast_to(i, (8, 50)).copy()
    for j in range(8):
        i[j,:] = i[j,:] + j*10
    r = getFeatures(i, [timeDomainStats], 10)
    print(r.shape) # (160,)
    r = getFeatures(i, [logMomentsFourierSpectra])
    print(r.shape) #(136,)
    r = getFeatures(i, [localBinaryPatterns])
    print(r.shape) # (16,)
    r = getFeatures(i, [spectralBandPowers])
    print(r.shape) #(8*NBANDS,)