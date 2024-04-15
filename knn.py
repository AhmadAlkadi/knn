#-------------------------------------------------------------------------
# AUTHOR: Ahmad Alkadi
# FILENAME: knn
# SPECIFICATION: knn prediction
# FOR: CS 5990- Assignment #3
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
#reading the test data
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
dataSets = ['weather_training.csv', 'weather_test.csv']
dfOne = pd.read_csv(dataSets[0])
data_training = np.array(dfOne.values)[:,1:].astype('f')
dfTwo = pd.read_csv(dataSets[1])
data_test = np.array(dfTwo.values)[:,1:].astype('f')
x_training = data_training[:,:-1]
y_training = data_training[:,-1]
y_training_normailzed = preprocessing.normalize([y_training])
y_training_midpoint = (max(y_training)-min(y_training))/2
y_training = np.where(y_training<y_training_midpoint,'low', 'high')
x_test = data_test[:,:-1]
y_test = data_test[:,-1]
y_test_normailzed = preprocessing.normalize([y_test])
y_test_midpoint = (max(y_test)-min(y_test))/2
y_test = np.where(y_test<y_test_midpoint,'low', 'high')

#loop over the hyperparameter values (k, p, and w) ok KNN
#--> add your Python code here
accuracy_high = 0
for k in k_values:
    for p in p_values:
        for w in w_values:

            #fitting the knn to the data
            #--> add your Python code here

            #fitting the knn to the data
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(x_training, y_training)

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            #--> add your Python code here
            count =0
            for (x_testSample, y_testSample) in zip(x_test, y_test):
                class_predict = clf.predict([x_testSample])
                if class_predict == y_testSample:
                    count+=1
            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            #--> add your Python code here
            accuracy_test = count / len(x_test)
            if(accuracy_test > accuracy_high):
                accuracy_high = accuracy_test
                print("Highest KNN accuracy so far: " + str(accuracy_high) + ", Parameters: k=" + str(k) + ", p=" + str(p) + ", w=" + str(w))




