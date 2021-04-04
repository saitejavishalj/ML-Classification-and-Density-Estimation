"""
Logistic Regression Model Implementation:
    This file has functions required for the implementation of Logistic Regression algorithm using Gradient Acsent. 
    Features are extracted here initially, the logistic regression model is trained and accuracy is calculated in the end. 
    Every function here is utilized in main.py.

"""
import scipy.io as sio
import numpy as np

"""
Mean and Standard deviation are the features calculated for the dataset, and 1's are attached along the column of the np array
"""
def features_logistic(inp):
    res = np.vstack( [np.ones(len(inp)), inp.mean(axis=1),inp.std(axis=1)] ).T ##Transpose the matrix
    return res


"""
Sigmoid function is calculated to get down all the values between 0 and 1, since the probability values are defined to be in between 0 and 1.
This can be further used to calculate the gradients.
"""
def calc_sigmoid(inp):
    return (np.exp(inp)) / (1 + np.exp(inp))

"""
The below function will train model using Logistic Regression Algorithm, and the weights are returned in here. 
These weights are used to predict the input set; utilizes Gradient Acsent Algorithm.
"""
def log_train(train_x, train_y, iter=100000, LR=0.01):
    train_x = features_logistic(train_x)
    logistic_weights = np.random.rand(3) # Random intialization
    
    for i in range(iter):
        weight_x = np.dot(train_x, logistic_weights) # perform dot product of Np Arrays
        normalized_vals = calc_sigmoid(weight_x) # Normalizing the values. 
        gradient = np.dot(train_x.T, (train_y[0] - normalized_vals))
        logistic_weights += LR * gradient 
    return logistic_weights

"""
The accuracy is calculated in the below function. 
Training labels are compared with the predicted labels.
"""
def calc_accuracy_log(logistic_result,test_y):
    acc = 0
    for i in range(len(logistic_result)):
        if (logistic_result[i]==test_y[i].T):
            acc = acc+1 
    acc = acc/len(test_y)
    return acc
