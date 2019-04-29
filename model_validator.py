import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import load_model
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

def readFiles(folder):
    test_x = np.genfromtxt(folder + "/test_x.csv", delimiter='\t', skip_header=True)[:, 1:]
    test_y = np.loadtxt(folder + "/test_y.csv", delimiter='\t', usecols=range(2)[1:], skiprows=1)
    
    test_ids = pd.read_csv(folder + "/test_y.csv", delimiter="\t", index_col=0, low_memory=False).index

    experiment.log_data_ref(data=test_x, data_name='test_x')
    experiment.log_data_ref(data=test_y, data_name='test_y')

    return test_x, test_y, test_ids

def readModelPaths(path):
    print("Reading model path: ", path)
    models = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for model in f:
            models.append(os.path.join(r, model))
    print("Got ", len(models), " clusters: ", models)
    return models

def scaleVectors(test_x):
    seed = 7
    np.random.seed(seed)
    sc = StandardScaler()
    scaled_test_x = sc.fit_transform(test_x)
    return scaled_test_x

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

    experiment.log_metrics(CA=CA)
    experiment.log_metrics(CR=CR)
    experiment.log_metrics(FA=FA)
    experiment.log_metrics(FR=FR)

    Correct = CA + FR
    Incorrect = CR + GFA + PFA
    Df = 0
    if (( CR + FA ) > 0 and CR > 0):
        IncorrectRejectionRate = CR / ( CR + FA )
    else:
        IncorrectRejectionRate = 'undefined'

    if (( FR + CA ) > 0 and FR > 0):
        CorrectRejectionRate = FR / ( FR + CA )
    else:
        CorrectRejectionRate = 'undefined'

    if ( CorrectRejectionRate != 'undefined' and IncorrectRejectionRate != 'undefined' and  CorrectRejectionRate != 0) :
        D = IncorrectRejectionRate / CorrectRejectionRate 
        experiment.log_metrics(D=D)
        # Further metrics
        Z = CA + CR + FA + FR
        Ca = CA / Z
        Cr = CR / Z
        Fa = FA / Z
        Fr = FR / Z

        P = Ca / (Ca + Fa)
        R = Ca / (Ca + Fr)
        SA = Ca + Cr
        F = (2 * P * R)/( P + R)
        
        RCa = Ca / (Fr + Ca)
        RFa = Fa / (Cr + Fa)
        
        print(D)    
        Da = RCa / RFa

        if ( D != 'undefined' ) :
            Df = math.sqrt((Da*D))
            experiment.log_metrics(Df=Df)
        else:
            Df = 'undefined'
    else:
        D = 'undefined'

    return Df

def testClassifier(classifier, scaled_test_x, test_y, test_ids):
    test_y_pred = classifier.predict_classes(scaled_test_x)
    prediction = dict(zip(test_ids, test_y_pred.flatten()))
    reality = dict(zip(test_ids, test_y))
    return prediction, reality

# Run dat naow
experiment = Experiment()

fullReality = dict()
fullPrediction = dict()

for model in readModelPaths('/data/shared-task/berkvec-models'):
    cluster = model.rsplit('/', 1)[-1].rsplit('.', 1)[0]

    test_x, test_y, test_ids = readFiles('/data/shared-task/berkvec/' + cluster)
    scaled_test_x = scaleVectors(test_x)
    classifier = load_model(model)

    # Test
    prediction, reality = testClassifier(classifier, scaled_test_x, test_y, test_ids.values)
    fullReality.update(reality)
    fullPrediction.update(prediction)

evaluate(list(fullReality.values()), list(fullPrediction.values()))