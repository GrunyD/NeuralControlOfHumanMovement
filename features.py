import numpy as np
from scipy import stats, signal

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
    


def getFeatures(emgData: np.ndarray, featureCalculationFuncs:list, segmentWindowLength:int = None) -> np.ndarray:
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
    i = np.broadcast_to(i, (8, 50))
    r = getFeatures(i, [timeDomainStats], 10)
    print(r.shape) # (160,)
