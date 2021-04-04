"""
Naive Bayes Model Implementation:
    This file will contain the functions for feature extraction (Standard deviation, mean), calculating the probabilities and Accuracy.
"""

import numpy as np

"""
    Get features from training dataset.
    Standard Deviation and Mean are the features selected to distiguish shirts and trousers.
"""
def features_naive_bayes(train_x,test_x):
    # Stacking standard deviation and mean for both test and train.
    train_x = np.vstack( [train_x.mean(axis=1),train_x.std(axis=1)] ).T  ## .T will give the transpose of the np Array
    test_x = np.vstack( [test_x.mean(axis=1),test_x.std(axis=1)] ).T
    ## train_x and test_x will have training and testing features.

    return train_x, test_x, train_x[0:6000],train_x[6000:],test_x[0:1000],test_x[1000:]


"""
The below function will contain initial probabilities for the Shirt and Trouser from Fashion MNIST dataset.
"""
def calc_prob_nb(train_y):
    variable, pre_probability = np.unique(train_y,return_counts=True)
    pre_probability = pre_probability/12000

    return pre_probability

"""
In terms of Gausian Normal Distribution, the likelihood is calculated in the below function. 
In short, P(X/Y) required for the Niave Bayes is calculated in the below function
"""
def calc_likelihood(X,m,variable):
    temp = 1/((2*np.pi*variable)**0.5) 
    res = temp * ( np.exp(-(X-m)**2 / (2*(variable)) ) ) 
    return res

"""
For the actual result of Naive Bayes, P(Y/X) - Post Probability is calculated in the below function.
.mean() function will return mean whereas .std() will return Standard Deviation.
0th column will have mean, 1st column will have standard deviation.
trx_updated_s - festures of Shirt is present.
trx_updated_t - features of Trouser is present.
"""
def nb_train(train_x,train_y,test_x):
    trx_updated, tsx_updated, trx_updated_s, trx_updated_t, tsx_updated_s, tsx_updated_t  = features_naive_bayes(train_x,test_x)
    temp_st = calc_prob_nb(train_y)
    post_prob_s = temp_st[0] * calc_likelihood(tsx_updated[:,0],trx_updated_s[:,0].mean(),trx_updated_s[:,0].var()) * calc_likelihood(tsx_updated[:,1],trx_updated_s[:,1].mean(),trx_updated_s[:,1].var())
    post_prob_t = temp_st[1] * calc_likelihood(tsx_updated[:,0],trx_updated_t[:,0].mean(),trx_updated_t[:,0].var()) * calc_likelihood(tsx_updated[:,1],trx_updated_t[:,1].mean(),trx_updated_t[:,1].var())
    
    NB_trained = np.zeros((len(post_prob_s)))
    for i in range(len(post_prob_s)):
        if post_prob_s[i]>post_prob_t[i]:
            NB_trained[i] = 0
        else:
            NB_trained[i] = 1

    return NB_trained

"""
Calculate accuracy of Naive Bayes through this function by comparing it with the labels in testing.
"""
def calc_accuracy_nb(NB_trained,test_y):
    acc = 0
    for i in range(len(NB_trained)):
        if (NB_trained[i]==test_y[i].T):
            acc = acc+1     
    acc = acc/len(test_y)

    return acc




