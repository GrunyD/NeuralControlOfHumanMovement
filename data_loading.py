import os
from os import path
import yaml
import numpy as np
from enum import IntEnum, auto

SAMPLING_RATE = 25
#TODO Put sampling rate only at one place

Activities = IntEnum('Activities', ['KneeExt', 'KneeFlex', 'DorsiFlex', 'PlamtarFlex',
                                    'Rest_lying', 'Rest_sitting', 'Standing', 'Standing_Support',
                                    'Cartoon', 'Treadmill_self', 'Treadmill_fast', 'Treadmill_fixed',
                                    'Treadmill_run', 'Floor_self', 'Floor_fast', 'Floor_run', 
                                    'Stair_up', 'Stair_down', 'Jump', 'Football'])

#TODO Decide which activities are considered intense
class IntenseActivities(IntEnum):
    ...

class IntermidiateActivities(IntEnum):
    ...

class RestingActivity(IntEnum):
    ...


GRUNY_DATA_LOCATION = "/Users/davidgrundfest/Desktop/DTU school stuff/Neural control of human movement/project_code/"
#hej

FILENAME_BASE = "Participant"
ACTIVITY_INTENSITY_LABELS = {}
ACTIVITY_LABELS = {}

def getDataFromFiles(recordingPath:str, configFilePath:str) -> tuple[np.ndarray, dict]:
    recording = np.genfromtxt(recordingPath, delimiter=',')
    with open(configFilePath) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return recording, config

def parseRecording(emgData:np.ndarray, config:dict) -> tuple[list, list]:
    """
    Extracts activity emg recordings according to config file and assing labels.

    Parameters:
    ------------
    - emgData: (samples, channels)
    - config

    Returns:
    - emgRecordings: list<np.ndarrays>, shape: (channels, samples)
    - labels: list<int>, labels corresponding to each activity
    """
    recordings = []
    labels = []
    for activity in [activityName for activityName in dir(Activities) if not activityName.startswith('__')]:
        try:
            start = config[F"{activity}_T1"] * SAMPLING_RATE
            end = config[F"{activity}_T2"] * SAMPLING_RATE
            recordings.append(emgData[start : end, :].T)
            labels.append(Activities[activity].value)

        except KeyError:
            for i in range(1, 4):
                try:
                    start = config[F"{activity}{i}_T1"] * SAMPLING_RATE
                    end = config[F"{activity}{i}_T2"] * SAMPLING_RATE
                    recordings.append(emgData[start : end, :].T)
                    labels.append(Activities[activity].value)
                except KeyError:
                    break

    return recordings, labels

def loadData(recordingsPath:str, configFilesPath:str, participantsNums:list|tuple|np.ndarray = None) -> tuple[list, np.ndarray]:
    """
    Parameters:
    -----------
    recordingsPath: Path to csv files containg emg recordings
    configFilesPath: Path to yaml files conating timestamps
    participantsNums: Numbers of participatns that we want to load

    Returns:
    -----------
    TODO
    """
    
    if participantsNums is None:
        participantsNums = [i for i in range(1, 25)]

    recordings = []
    labels = []
    for pNum in participantsNums:
        recordingPath = path.join(recordingsPath, F"{FILENAME_BASE}{pNum}.csv")
        configFilePath = path.join(configFilesPath, F"{FILENAME_BASE}{pNum}.yaml")
        recording, config = getDataFromFiles(recordingPath, configFilePath)

        distinctRecordings, distinctLabels = parseRecording(recording, config)
        recordings.extend(distinctRecordings)
        labels.extend(distinctLabels)

    # labels = np.array(labels)
    return recordings, labels

if __name__ == "__main__":
    recordings, labels = loadData(F"{GRUNY_DATA_LOCATION}data", F"{GRUNY_DATA_LOCATION}yaml_extraction_files", [1, 2])
    print(len(recordings)) #50
    print(len(labels)) #50
