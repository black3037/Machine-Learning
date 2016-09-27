from __future__ import division
import numpy as np
"""
Copyright (C) Derek Black 2016

Required Modules:
Numpy

Topics include:
Gradient Decent
Logistic Regression
Cost Function
Sigmoid Function


This module is for multi-variant problems.
Gradient Decent to optimize the cost function to automatically find the 
optimized thetas for 
a minimized cost function.

How to use LogisticRegression:
LogisticRegression is an iterative method for solving logistic regression problems.

LogisticRegression utilizes a method called Gradient Decent to find values of theta
for a minimized cost function. The function returns the optimized values of theta.

Ex. theta = LogisticRegression([1,2,3],[10,12,15],1500,0.4)

"""

# Compute the cost function
def costfunction(x, y, theta):
    
    m = x.shape[0] # Get shape of training examples
    
    sigmoid = (1/(1+np.exp(x.dot(theta)))) # Sigmoid function for logistic Regression
    
    # Compute the cost and return
    cost = (1./(m))*(-y.T.dot(np.log(sigmoid))-(1-y).T.dot(np.log(1-sigmoid)))
    
    return cost.flat[0]

def LogisticRegression(x, y, theta, iteration, alpha):
    theta_iter = [] #record theta for each iteration
    cost_iter = []  #record cost for each iteration
    m = x.shape[0]


    for i in range(iteration):
        sigmoid = (1/(1+np.exp(x.dot(theta))))
        
        # Compute current value of theta and append to list
        theta = theta-(alpha/m)*x.T*(sigmoid-y)
        theta_iter.append(theta)
        
        # Append cost function list
        cost_iter.append(costfunction(x,y,theta))

    return theta
    
# Credit goes to Marcel Caraciolo for the map feature function
def map_feature(x1, x2):
    '''
    Maps the two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
    Inputs X1, X2 must be the same size
    '''
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = np.ones(shape=(x1[:, 0].size, 1))

    m, n = out.shape

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)

    return out