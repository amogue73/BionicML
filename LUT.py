import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier

def define_LUT():

    LUT = np.zeros(MAX_FEATURES)
    inputX = np.zeros((NUM_SAMPLES_TRAIN,1), dtype=float)
    inputY = np.zeros((NUM_SAMPLES_TRAIN,), dtype=int)
    testX = np.zeros((NUM_SAMPLES_TEST,1), dtype=float)
    testY = np.zeros((NUM_SAMPLES_TEST,), dtype=int)

    for i in range(NUM_SAMPLES_TRAIN):
        inputY[i] = TRAIN_Y[i]

    for i in range(NUM_SAMPLES_TEST):
        testY[i] = TEST_Y[i]
    
    for i in range(MAX_FEATURES):
        neigh = KNeighborsClassifier(n_neighbors=100)
        for j in range(NUM_SAMPLES_TRAIN):
            inputX[j][0] = TRAIN_X[j][i]

        for j in range(NUM_SAMPLES_TEST):
            testX[j][0] = TEST_X[j][i]

        neigh.fit(inputX,inputY)

        n_success = 0
    
        prediction = neigh.predict(testX)
        
        for j in range(NUM_SAMPLES_TEST):
            if (prediction[j] == testY[j]):
                n_success += 1
    
        acc = n_success/NUM_SAMPLES_TEST
        if (acc < 1/NUM_CHAR):
            LUT[i] = 0
        else:
            LUT[i] = acc - 1/(NUM_CHAR -1)*(1 - acc)
    return LUT

TRAIN_X = []
TRAIN_Y = []
TEST_X = []
TEST_Y = []

with open('Essex/104_training_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        TRAIN_X.append([float(feature) for feature in row])

with open('Essex/104_training_class.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        TRAIN_Y.append(int(row[0]))

with open('Essex/104_testing_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        TEST_X.append([float(feature) for feature in row])

with open('Essex/104_testing_class.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        TEST_Y.append(int(row[0]))

NUM_CHAR = 3 # number of different classes
MAX_FEATURES = 3600 # total number of features

NUM_SAMPLES_TRAIN = len(TRAIN_X) # number of samples used for training
NUM_SAMPLES_TEST = len(TEST_X) # number of samples used for testing

LUT = define_LUT()
print(len(LUT))

with open("LUT.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(LUT)