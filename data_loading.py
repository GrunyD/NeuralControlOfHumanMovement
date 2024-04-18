import os
from os import path
import yaml
import numpy as np
from enum import IntEnum
import random



"""
Expects csv files with the emg recordings, where rows are samples and columns are channels
Expects yaml files that specify when each trial begins and stops
The keys in yaml file are in this format: {Activity}{number of trial for this activity:optional}_{T1 or T2}
"""

SAMPLING_RATE = 25
#TODO Put sampling rate only at one place

Activities = IntEnum('Activities', [#'KneeExt', 'KneeFlex', 'DorsiFlex', 'PlamtarFlex',
                                    'Rest_lying', 'Rest_sitting', 'Standing', 'Standing_Support',
                                    'Cartoon', 'Treadmill_self', 'Treadmill_fast', 'Treadmill_fixed',
                                    'Treadmill_run', 'Floor_self', 'Floor_fast', 'Floor_run', 
                                    'Stair_up', 'Stair_down', 'Jump', 'Football'])


IntenseActivities = IntEnum('IntenseActivities', ['Jump', 'Football', 'Floor_fast', 'Floor_run', 'Treadmill_run', 'Treadmill_fast', 'Stair_up'] )

IntermediateActivities = IntEnum('IntermediateActivities', ['Stair_down', 'Floor_self','Treadmill_self', 'Treadmill_fixed'])

RestingActivities = IntEnum('RestingActivities', ['Cartoon', 'Rest_sitting', 'Rest_lying','Standing', 'Standing_Support'])


#GRUNY_DATA_LOCATION = "/Users/davidgrundfest/Desktop/DTU school stuff/Neural control of human movement/project_code/"
Sofie_DATA_LOCATION = "C:\\Users\\sofie\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\DTU\\8. semester\\Biomechanics"
Anton_DATA_LOCATION = "C:/Users/anton/OneDrive/Skrivebord/Biomechanics and neural control of movements"

FILENAME_BASE = "Participant"

def getDataFromFiles(recordingPath:str, configFilePath:str) -> tuple[np.ndarray, dict]:
    recording = np.loadtxt(recordingPath, delimiter=',')
    with open(configFilePath) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError
    return recording, config

def getLabel(activity, type:str = 'intensity'):
    """
    Returns a label we want to use in classification. There are two types of tasks. Distinguishing
    between activities and then distinguishing between intensities in the activities.

    Parameters:
    ------------
    -   activity: enum attribute
    -   type: specifies the type of the task, intensity or activity.
            Defaults to intesity

    Returns:
    -----------
    -   label: int

    Raises:
    -----------
    -   NotImplementedError: if uses not implemented type
    """
    if type == 'intensity':
        for i, activity_enum in enumerate([IntenseActivities, IntermediateActivities, RestingActivities]):
            if activity in dir(activity_enum):
                return i
    elif type == 'activity':
        return Activities[activity].value
    else:
        raise NotImplementedError(F'This function doesnt take this "{type}" as type')


def parseRecording(emgData:np.ndarray, config:dict, augmentation_dict:dict = None, limit:int = 10) -> tuple[list, list]:
    """
    Extracts activity emg recordings according to config file and assing labels.

    Parameters:
    ------------
    - emgData: (samples, channels)
    - config: dictionary of timestamps for trials
    - augmentation_dict: contains info about possible augmentation; if None, no augmentatio is done

    Returns:
    -----------
    - emgRecordings: list<np.ndarrays>, shape: (channels, samples)
    - labels: list<int>, labels corresponding to each activity
    """
    recordings = []
    labels = []
    for activity in [activityName for activityName in dir(Activities) if not activityName.startswith('__')]:
        # Not all activities are in the same format, some of them occured only once, 
        # some of them several time
        # We have to get starting end ending points of those activities
        starts = []
        ends = []
        try:
            starts.append(config[F"{activity}_T1"] * SAMPLING_RATE)
            ends.append(config[F"{activity}_T2"] * SAMPLING_RATE)

        except KeyError: #If the activity has more trials, then it iterates through those trials
            for i in range(1, 4):
                try:
                    starts.append(config[F"{activity}{i}_T1"] * SAMPLING_RATE)
                    ends.append(config[F"{activity}{i}_T2"] * SAMPLING_RATE)
                except KeyError:
                    break
        # We have really low number of data
        # We perform time shifts augmentation so it is more robust towards 
        # random timeshifts in real world measuring
        if augmentation_dict is not None:
            rng = augmentation_dict.get('range', SAMPLING_RATE//2)
            n_samples = augmentation_dict.get('n_samples', 20)
            shifts = np.random.randint(low = -rng, high = rng+1, size = n_samples)
        else:
            shifts = [0,]

        assert len(starts) == len(ends), F"{len(starts)}{len(ends)}"
        for start, end in zip(starts, ends):
            for shift in shifts:
                recording = emgData[start + shift: end + shift, :].T
                Nsegments = recording.shape[1]//SAMPLING_RATE
                splitRecording = np.array_split(recording, Nsegments, axis = 1)
                if limit is not None and len(splitRecording) > limit:
                    random.choices(splitRecording, k = limit)
                recordings.extend(splitRecording)
                labels.extend([getLabel(activity) for _ in range(Nsegments)])
                assert len(recordings) == len(labels),F"Labels are not the same length after recording segmntation: {len(recordings)}, {len(labels)}"
            
    return recordings, labels

def loadData(recordingsPath:str, configFilesPath:str, participantsNums:list|tuple|np.ndarray = None, augmentation_dict:dict = None) -> tuple[list, np.ndarray]:
    """
    Parameters:
    -----------
    recordingsPath: Path to csv files containg emg recordings
    configFilesPath: Path to yaml files conating timestamps
    participantsNums: Numbers of participatns that we want to load

    Returns:
    -----------
    recordings: list of numpy arrays containing the emg recordings
    labels: list of ints
    """
    
    if participantsNums is None:
        participantsNums = [i for i in range(1, 25)]

    recordings = []
    labels = []
    for pNum in participantsNums:
        print(F'Loading participant {pNum}')
        recordingPath = path.join(recordingsPath, F"{FILENAME_BASE}{pNum}.csv")
        configFilePath = path.join(configFilesPath, F"{FILENAME_BASE}{pNum}.yaml")
        recording, config = getDataFromFiles(recordingPath, configFilePath)

        distinctRecordings, distinctLabels = parseRecording(recording, config, augmentation_dict)
        recordings.extend(distinctRecordings)
        labels.extend(distinctLabels)

    # labels = np.array(labels)
    return recordings, labels

if __name__ == "__main__":
    # GRUNY_DATA_LOCATION = "/Users/davidgrundfest/Desktop/DTU school stuff/Neural control of human movement/project_code/"
    # Data_location = os.
    dirname, filename = os.path.split(__file__)
    recordings, labels = loadData(os.path.join(dirname, 'data'), 
                                  os.path.join(dirname, 'Participants_time_stamps_data'),
                                  [1, 2],
                                  {'range':20, 'n_samples': 20})
    print(len(recordings)) #50
    print(recordings[0].shape)
    print(len(labels)) #50
    


