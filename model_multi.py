import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import math
from keras.utils.vis_utils import plot_model
import uuid
from polyaxon_client.tracking import Experiment, get_log_level, get_data_paths, get_outputs_path
from polyaxon_client.tracking.contrib.keras import PolyaxonKeras
import argparse
import os

def readClusterFoldersFromBaseFolder(path):
    print("Reading path: ", path)
    folders = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for folder in d:
            folders.append(os.path.join(r, folder))
    print("Got ", len(folders), " clusters: ", folders)
    return folders

def readFiles(folder):
    train_x = np.genfromtxt(folder + "/train_x.csv", delimiter='\t', skip_header=True)[:, 1:]
    train_y = np.loadtxt(folder + "/train_y.csv", delimiter='\t', usecols=range(2)[1:], skiprows=1)
    
    test_x = np.genfromtxt(folder + "/test_x.csv", delimiter='\t', skip_header=True)[:, 1:]
    test_y = np.loadtxt(folder + "/test_y.csv", delimiter='\t', usecols=range(2)[1:], skiprows=1)
    
    test_ids = pd.read_csv(folder + "/test_y.csv", delimiter="\t", index_col=0, low_memory=False).index
    return train_x, train_y, test_x, test_y, test_ids

def scaleVectors(train_x, test_x):
    seed = 7
    np.random.seed(seed)
    sc = StandardScaler()
    scaled_train_x = sc.fit_transform(train_x)
    scaled_test_x = sc.transform(test_x)
    return scaled_train_x, scaled_test_x

def trainClassifier(scaled_train_x, train_y):
    
    # Hyperparam
    batch_size = 200
    learning_rate = 0.05
    dropout = 0.2
    num_epochs = 200
    
    # InputSize
    input_dim = len(train_x[0])
    layer_dim = max(input_dim, 64)
    
    # Structure
    classifier = Sequential()
    classifier.add(Dense(layer_dim, activation='relu', input_dim=input_dim))
    classifier.add(Dense(layer_dim, activation='relu'))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(1, activation='sigmoid'))
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    classifier.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    metrics = classifier.fit(scaled_train_x, train_y, batch_size = batch_size, epochs = num_epochs, validation_split=0.1)
    return classifier

def evaluate(true_y, pred_y):
    true_classes = true_y
        
    CR, CA, PFA, GFA, FR, k = 0, 0, 0, 0, 0, 3.0
    for idx, prediction in enumerate(pred_y):
        # the students answer is correct in meaning and language
        # the system says the same -> accept
        if true_classes[idx] == 1 and prediction == 1:
            CA += 1
        # the system says correct meaning wrong language -> reject
        elif true_classes[idx] == 1 and prediction == 0:
            FR += 1

        # students answer is correct in meaning and wrong in language
        #The system says the same -> reject
        elif true_classes[idx] == 0 and prediction == 0:
            CR += 1
        # the system says correct meaning and correct language -> accept
        elif true_classes[idx] == 0 and prediction == 1:
            PFA += 1

    FA = PFA + k * GFA
    Correct = CA + FR
    Incorrect = CR + GFA + PFA
    IncorrectRejectionRate = CR / ( CR + FA + 0.0 )
    CorrectRejectionRate = FR / ( FR + CA + 0.0 )
    # Further metrics
    Z = CA + CR + FA + FR
    Ca = CA / Z
    Cr = CR / Z
    Fa = FA / Z
    Fr = FR / Z
    print("CA:",CA)
    print("CR:",CR)
    print("FA:",FA)
    print("FR:",FR)

    P = Ca / (Ca + Fa)
    R = Ca / (Ca + Fr)
    SA = Ca + Cr
    F = (2 * P * R)/( P + R)
    
    RCa = Ca / (Fr + Ca)
    RFa = Fa / (Cr + Fa)
    
    D = IncorrectRejectionRate / CorrectRejectionRate
    print(D)
    Da = RCa / RFa
    Df = math.sqrt((Da*D))
    return Df

def testClassifier(classifier, scaled_test_x, test_y, test_ids):
    test_y_pred = classifier.predict_classes(scaled_test_x)
    prediction = dict(zip(test_ids, test_y_pred.flatten()))
    reality = dict(zip(test_ids, test_y))
    return prediction, reality

# Run dat naow

clusterFolders = readClusterFoldersFromBaseFolder('/data/shared-task/berkvec/')

fullReality = dict()
fullPrediction = dict()

for folder in clusterFolders:
    train_x, train_y, test_x, test_y, test_ids = readFiles(folder)
    scaled_train_x, scaled_test_x = scaleVectors(train_x, test_x)
    classifier = trainClassifier(scaled_train_x, train_y)
    prediction, reality = testClassifier(classifier, scaled_test_x, test_y, test_ids.values)
    fullReality.update(reality)
    fullPrediction.update(prediction)

evaluate(list(fullReality.values()), list(fullPrediction.values()))