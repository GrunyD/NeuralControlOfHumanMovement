from sklearn.neighbors import KNeighborsClassifier
from data_loading import loadData
from features import getFeatures
import os
import numpy as np
from tqdm import tqdm

AUGMENTATION_SAMPLES = 5
AUGMENTATION_RANGE = 14

# KNN and SVM
def getData(participantsList:list, augmentation:bool = True):
    # Load data
    print('Loading Data')
    dirname, filename = os.path.split(__file__)
    dataFolder = os.path.join(dirname, 'data')
    configFolder = os.path.join(dirname, 'Participants_time_stamps_data')

    recordings, y = loadData(dataFolder, 
                    configFolder, 
                    participantsList, 
                    #{'range': AUGMENTATION_RANGE, 'n_samples': AUGMENTATION_SAMPLES}
                    )
    

    #Extract features
    print('Extracting features')

    x  = [getFeatures(recording) for recording in recordings]

    # Format data in nice way for classifiers
    y = np.array(y)
    x = np.array(x)

    return x, y

def accuracy(classifier, X, y) -> float:
    y_hat = classifier.predict(X)
    result = (y_hat == y)
    return np.sum(result)/len(result)

if __name__ == '__main__':
    trainParticipants = [i for i in range(5, 26)]
    valParticipants = [1, 2, 3, 4]
    X_train, y_train = getData(trainParticipants, augmentation=True)
    X_val, y_val = getData(valParticipants, augmentation=False)
    # print(np.unique(y_val))
    classifiers = [KNeighborsClassifier(n_neighbors=5)]
    for classifier in classifiers:
        classifier.fit(X_train, y_train)
        print('___________________________________________')
        print('Classifier: ', classifier)
        print('Train accuracy: ', accuracy(classifier, X_train, y_train))
        print('Validation accuracy: ', accuracy(classifier, X_val, y_val))
        print('\n')


#TODO what is nperseg
