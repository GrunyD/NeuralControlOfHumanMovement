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

# Activities = IntEnum('Activities', [#'KneeExt', 'KneeFlex', 'DorsiFlex', 'PlamtarFlex',
#                                     'Rest_lying', 'Rest_sitting', 
#                                     'Standing', 'Standing_Support',
#                                     # 'Cartoon', 
#                                     'Treadmill_self', 'Floor_self', 
#                                     'Floor_fast','Treadmill_fast', 
#                                     'Treadmill_fixed',
#                                     'Treadmill_run', 'Floor_run', 
#                                     'Stair_up', 'Stair_down',
#                                     'Jump', 'Football'
#                                     ])

# All activities
Activities = [#'KneeExt', 'KneeFlex', 'DorsiFlex', 'PlamtarFlex',
            'Rest_lying', 'Rest_sitting', 
            'Standing', 'Standing_Support',
            # 'Cartoon', 
            'Treadmill_self', 'Floor_self', 
            'Floor_fast','Treadmill_fast', 
            'Treadmill_fixed',
            'Treadmill_run', 'Floor_run', 
            'Stair_up', 'Stair_down',
            # 'Jump', 'Football'
            ]


# Intensity division
IntenseActivities = ['Jump', 'Football', 'Floor_fast', 'Floor_run', 'Treadmill_run', 'Treadmill_fast', 'Stair_up']
IntermediateActivities = ['Stair_down', 'Floor_self','Treadmill_self', 'Treadmill_fixed']
RestingActivities = ['Cartoon', 'Rest_sitting', 'Rest_lying','Standing', 'Standing_Support']

# Merged activities division
Lying = ['Rest_lying',]
Sitting = ['Rest_sitting',]
Standing = ['Standing', 'Standing_Support',]
Walking = ['Treadmill_self', 'Floor_self',]
Fixed_walking = ['Treadmill_fixed']
Fast_walking = ['Floor_fast', 'Treadmill_fast']
Running = ['Floor_run', 'Treadmill_run']
Stairs_up = ['Stair_up']
Stairs_down = ['Stair_down']
merged_activities = ['Lying', 'Sitting','Standing', 'Walking', 'Fast_walking', 'Fixed_walking', 'Running', 'Stairs_up', 'Stairs_down']





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

def getLabel(activity_index:int, type:str = 'activity'):
    """
    Returns a label as number we want to use in classification. There are two types of tasks. Distinguishing
    between activities and then distinguishing between intensities in the activities.

    Parameters:
    ------------
    -   activity_index: int
    -   type: specifies the type of the task, 
            - intensity
            - activity
            - merged_activities
            Defaults to activity

    Returns:
    -----------
    -   label: int

    Raises:
    -----------
    -   NotImplementedError: if uses not implemented type
    """
    if type == 'intensity':
        for i, activities_list in enumerate([IntenseActivities, IntermediateActivities, RestingActivities]):
            if Activities[activity_index] in activities_list:
                return i
    elif type == 'activity':
        return activity_index
    
    elif type == 'merged_activities':
        for i, activities_list in enumerate([Lying, Sitting, Standing, Walking, Fast_walking, Fixed_walking, Running, Stairs_up, Stairs_down]):
            if Activities[activity_index] in activities_list:
                return i
    else:
        raise NotImplementedError(F'This function doesnt take this "{type}" as type')
    

def getLabelName(label:int, task_type:str)->str:
    if task_type == 'activity':
        return Activities[label]
    elif task_type == 'intensity':
        return ['High Intensity', 'Intermediate Intensity', 'Resting'][label]
    elif task_type == 'merged_activities':
        a = merged_activities[label]
        return a
    else:
        raise NotImplementedError

def parseRecording(emgData:np.ndarray, config:dict, augmentation_dict:dict = None, limit:int = 10, length:int = 50, overlap:float = 0.5, task_type:str = 'intensity') -> tuple[list, list]:
    """
    Extracts activity emg recordings according to config file and assing labels.

    Parameters:
    ------------
    - emgData: (samples, channels)
    - config: dictionary of timestamps for trials
    - augmentation_dict: contains info about possible augmentation; if None, no augmentatio is done
        it takes:   time_range (when doing time_shift augmentation)
                    n_smaples (when doing time_shift augmentation)
                    scaling_range (taking amplitude scaler from uniform distribution with this range)
    - limit: we divide each trial into segments of the same lengths, some trials are way longer and thus have way more samples, 
            this limits number of segments from a trial
    - length: When spliting the whole recording into smaller recordings, how long the smaller recordings should be (in samples)
    - overlap: When spliting the whole recording into smaller recordings, how much should they overlap, 
            you can not have 100% overlap, in that case, sthe stride is set to 1,
            if it is <= 0, stride is bigger than length and there are jumps between recordings
            defaults to 0.5
    - task_type: to pass to getLabel

    Returns:
    -----------
    - emgRecordings: list<np.ndarrays>, shape: (channels, samples)
    - labels: list<int>, labels corresponding to each activity
    """
    recordings = []
    labels = []
    # for activity in [activityName for activityName in dir(Activities) if not activityName.startswith('__')]:
    for activity_index, activity in enumerate(Activities):
        # Not all activities are in the same format, some of them occured only once, 
        # some of them several time
        # We have to get starting and ending points of those activities
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

            if not isinstance(augmentation_dict, dict): # To not throw error and simple setting of values
                augmentation_dict = {}

            rng = augmentation_dict.get('time_range', SAMPLING_RATE//2)
            n_samples = augmentation_dict.get('n_samples', 2)
            shifts = np.random.randint(low = -rng, high = rng+1, size = n_samples)
            scaler_range = augmentation_dict.get('scaling_range', (0.95,1.05))
        else:
            shifts = [0,]
            scaler_range = (1,1)

        assert len(starts) == len(ends), F"{len(starts)}{len(ends)}"
        for start, end in zip(starts, ends):
            for shift in shifts:

                recording = emgData[start + shift: end + shift, :].T

                # We need to change to lenght of each recording from test to test
                # We need some overlap of the recordings for small sample activities (jumps)
                splitRecording = []
                start = 0
                stride = int(length * (1 - overlap)) # If the overlap is 0.6, then the stride should be 0.4 so there is bigger overlap
                if stride <= 0:
                    stride = 1

                while start + length <= recording.shape[1]:
                    scaler = random.uniform(*scaler_range) # For part of recording also chooses amplitude scaler
                    try:
                        splitRecording.append(scaler * recording[:, start:start+length])
                    except IndexError:
                        splitRecording.append(scaler * recording[:, start:]) # This should be the same length, but is able to go to the end
                        print(F"\x1b[1;36mDEBUG:\x1b[0;37m Last recording shape :{splitRecording[-1].shape}")
                    finally:
                        start += stride

                # Some recordings are super long and would completely outweight other data (resting)
                if limit is not None and len(splitRecording) > limit:
                    splitRecording = random.choices(splitRecording, k = limit)
 
                recordings.extend(splitRecording)
                labels.extend([getLabel(activity_index, type=task_type) for _ in range(len(splitRecording))])
                assert len(recordings) == len(labels),F"Labels are not the same length after recording segmntation: {len(recordings)}, {len(labels)}"
            
    return recordings, labels




def loadData(recordingsPath:str, configFilesPath:str, participantsNums:list|tuple|np.ndarray = None, augmentation_dict:dict = None, limit:int = 10, length:int = 50, overlap:float = 0.5, task_type:str = 'intensity') -> tuple[list, np.ndarray]:
    """
    Parameters:
    -----------
    - recordingsPath: Path to csv files containg emg recordings
    - configFilesPath: Path to yaml files conating timestamps
    - participantsNums: Numbers of participatns that we want to load

    Returns:
    -----------
    - recordings: list of numpy arrays containing the emg recordings
    - labels: list of ints
    """
    
    if participantsNums is None:
        participantsNums = [i for i in range(1, 26)]

    recordings = []
    labels = []
    for pNum in participantsNums:
        print(F'Loading participant {pNum}')
        recordingPath = path.join(recordingsPath, F"{FILENAME_BASE}{pNum}.csv")
        configFilePath = path.join(configFilesPath, F"{FILENAME_BASE}{pNum}.yaml")
        recording, config = getDataFromFiles(recordingPath, configFilePath)

        distinctRecordings, distinctLabels = parseRecording(recording, config, augmentation_dict, limit, length, overlap, task_type)
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
    


