# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:51:00 2021

@author: Sai Teja Vishal J

Notes: Please run the main file and make sure the other two files "nb_implemneted.py" and "lr_implemented.py" are in the
same folder of Main.py.
Thank you!
"""

import scipy.io as sio
import numpy as np

import nb_implemented as naiveB
import lr_implemented as logR

fdata = sio.loadmat('fashion_mnist.mat')

##Naive Bayes implemented :
NB_trained = naiveB.nb_train(fdata['trX'],fdata['trY'],fdata['tsX']) 
##Train based on Implemented Naive Bayes model to predict the test data. NB_trained is an array containing the predictions made by model.
## Implementation contained in nb_implemented.py

##Accuracies are calculated using the accuracy function in nb_implemented.py
print("Implemented Naive Bayes's accuracy for Testing Set - {:.4f}%".format(naiveB.calc_accuracy_nb(NB_trained, fdata['tsY'][0])*100))
print("Implemented Naive Bayes's accuracy for Shirt -  {:.4f}%".format(naiveB.calc_accuracy_nb(NB_trained[0:1000], fdata['tsY'][0][0:1000])*100))
print("Implemented Naive Bayes's accuracy for Trouser -  {:.4f}%".format(naiveB.calc_accuracy_nb(NB_trained[1000:], fdata['tsY'][0][1000:])*100))
##Accuracies for the testing set, Shirt and Trouser is printed here.
print("\n")


##Logistic Regression:

logistic_weights = logR.log_train(fdata['trX'], fdata['trY'], 89900, 0.003)
##Train the Logistic Regression model with the fashion data, the number of iterations are chosen and Learning Rate is also inputed here.
##Implementation contained in lr_implemented.py

#print(logistic_weights)
test_x = logR.features_logistic(fdata['tsX'])
logistic_result = logR.calc_sigmoid(np.dot(test_x,logistic_weights)) ##Use the logistis regression on test dataset, use sigoid function to normalize the values
logistic_result = (logistic_result>0.5).astype('int') ##if the result is greater than 0.5, then 1, else 0.
print("Implemented logistic regression's accuracy for Testing Set - {:.4f}%".format(logR.calc_accuracy_log(logistic_result,fdata['tsY'][0])*100))
logistic_result_s = logistic_result[0:1024] ##logistic_result_s will have results for Shirts using Logistic Regression.
logistic_result_t = logistic_result[1024:] ##logistic_result_t will have results for Shirts using Logistic Regression.
test_y_s = fdata['tsY'][0][0:1024] ##testing dataset for Shirts
test_y_t = fdata['tsY'][0][1024:] ##testing data for Trousers
#acc = (logistic_result == fdata['tsY'][0]).astype(int).sum()

##Accuracy is printed for Shirt and Trouser below, using the accuracy function implemented in lr_implemented.py
print("Implemented logistic regression's accuracy for Shirt - {:.4f}%".format(logR.calc_accuracy_log(logistic_result_s,test_y_s.T)*100))
print("Implemented logistic regression's accuracy for Trouser - {:.4f}%".format(logR.calc_accuracy_log(logistic_result_t,test_y_t.T)*100))
