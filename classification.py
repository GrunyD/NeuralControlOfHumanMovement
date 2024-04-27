from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from data_loading import loadData, getLabelName, Activities, merged_activities
from features import getFeatures
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sn

AUGMENTATION = False
AUGMENTATION_SAMPLES = 10
AUGMENTATION_TIME_RANGE = 14
AUGMENTATION_SCALING_RANGE = (1, 1)

LENGTH = 25
TASK_TYPE = 'activity'
LABELS = Activities

TD = np.array([4, 5, 9, 14, 15, 16, 17, 20, 21, 22, 24, 25])
CP = np.array([1, 2, 3, 6, 7, 8, 10, 11,12, 13, 18, 19, 23])



participants = [1,2]


# KNN and SVM
def getData(participantsList:list, augmentation:bool = True, task_type:str = 'intensity') -> tuple:
    # Load data
    print('Loading Data')
    dirname, _ = os.path.split(__file__)
    dataFolder = os.path.join(dirname, 'data')
    configFolder = os.path.join(dirname, 'Participants_time_stamps_data')

    augmentation_dict = None
    if augmentation:
        augmentation_dict = {'time_range': AUGMENTATION_TIME_RANGE, 
                             'n_samples': AUGMENTATION_SAMPLES, 
                             'scaling_range': AUGMENTATION_SCALING_RANGE}
        
    recordings, y = loadData(dataFolder, 
                    configFolder, 
                    participantsList, 
                    augmentation_dict,
                    limit = None,
                    task_type = task_type,
                    length = LENGTH
                    )
    
    #Extract features
    
    x = []
    for recording in tqdm(recordings, desc="Extracting features"):
        x.append(getFeatures(recording))

    # Format data in nice way for classifiers
    y = np.array(y)
    x = np.array(x)
    X_train = []
    X_val = []
    y_train = []
    y_val = []
    for label in np.unique(y):
        label_index = y == label
        label_features = x[label_index]
        label_labels = y[label_index]
        split_features = np.array_split(label_features, 5, 0)
        split_labels = np.array_split(label_labels, 5)
        X_val.append(split_features[0])
        X_train.extend(split_features[1:])
        y_val.append(split_labels[0])
        y_train.extend(split_labels[1:])

    #Split to training and val data
    # print([i.shape for i in X_train])
    return np.vstack(X_train), np.hstack(y_train), np.vstack(X_val), np.hstack(y_val)


def accuracy(y_hat, y) -> float:
    result = (y_hat == y)
    return np.sum(result)/len(result)

def crossvalidation(classifiers:list, subject_indices: list, k:int = 5) -> None:
    """
    Performs k-fold cross validation over the classifiers given in the list with patients specified
    Prints the results.

    Parameters:
    ------------
    -   classifiers: list of classifiers, they are expected to have methods fit and predict
    -   subject_indices: list of integers, specifying which subjects data will be used
    -   k: int, defines k-fold crossvalidation
    """
    print(subject_indices)
    classifier_results = [[] * len(classifiers)]
    subject_indices = np.array(subject_indices)
    np.random.shuffle(subject_indices)
    print(subject_indices)
    subsets = np.array_split(subject_indices, indices_or_sections=k) #shuffles and splits into k subsets
    
    for subset_index in range(k):
        print(F"\nStarting fold {subset_index + 1}")
        # Create the dataset for this fold
        np.random.seed(42)
        index = np.arange(k)[np.arange(k) != subset_index] # indices from the split that i want to use for training data
        trainParticipants = np.concatenate([subsets[i] for i in index]) # Takes the split array and put together the parts that were chosen for training data
        valParticipants = subsets[subset_index]
        X_train, y_train = getData(trainParticipants, augmentation=True)
        X_val, y_val = getData(valParticipants, augmentation=False)

        for index, classifier in enumerate(classifiers):
            print('___________________________________________')
            print('Classifier: ', classifier)
            classifier.fit(X_train, y_train)
            print('Train accuracy: ', accuracy(classifier, X_train, y_train))
            val_accuracy = accuracy(classifier, X_val, y_val)
            print('Validation accuracy: ', val_accuracy)
            # val_accuracy = 1/k
            classifier_results[index].append(val_accuracy) #append val_accuracy to list of val_accuracies for given classifier
            print('\n')

    print('\n\n___________________________________________')
    print("RESULTS:")
    for index, classifier in enumerate(classifiers):
        results = np.array(classifier_results[index])
        print('\nClassifier: ', classifier)
        print("\t\tAll validations: ", results)
        print("\t\tMean validation accuracy: ", np.mean(results))
        

if __name__ == '__main__':
    classifiers = [
        KNeighborsClassifier(n_neighbors=5), 
        # SVC(),
        ]
    for classifier in classifiers:
        print('___________________________________________')
        print('Classifier: ', classifier)
        target = []
        prediction = []
        for participant in participants:
            X_train, y_train, X_val, y_val = getData([participant,], augmentation=AUGMENTATION, task_type=TASK_TYPE)
            classifier.fit(X_train, y_train)
            target.extend([getLabelName(y, TASK_TYPE) for y in y_val])
            prediction.extend([getLabelName(y, TASK_TYPE) for y in classifier.predict(X_val)])


        target = np.array(target)
        prediction = np.array(prediction)
            
        print("\nActivity distribution in validation data")
        for label in np.unique(target):
            print(F"Label {label}: {int(100*np.sum(target == label)/len(target))}%")
        
        print('\nValidation accuracy: ', accuracy(prediction, target))

        
        m = confusion_matrix(target, prediction, labels= LABELS)
        sn.heatmap(m, annot=True, xticklabels=LABELS, yticklabels=LABELS)
        
        plt.show()

