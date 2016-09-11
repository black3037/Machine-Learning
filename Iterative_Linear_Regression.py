from __future__ import division
import random
import matplotlib.pyplot as plt

"""

Copyright Derek Black 2016

Part 1 of Machine Learning Package in Python

Topics include:
Gradient Decent
Linear Regression
Cost Functions

This module is for one varient iterative supervised machine learning problems.

"""

def hypothesis(t0,t1,x,y):
    return (t0 + t1*x - y)

# Adjust training elements (m) and learning rate (alpha) according to data
def gradient_decent(x,y,m=10,alpha=0.4):
    
    # Initial Conditions
    theta_0 = 0
    theta_1 = 0
    theta0 = []
    theta1 = []
    J = []
    
    # Iterate over range m to calculate theta1 and theta0
    for j in range(m):
        
        # calculate the gradient decent
        theta_0 = theta_0 - alpha*(1/m)*hypothesis(theta_0,theta_1,x[j],y[j])
        theta_1 = theta_1 - alpha*(1/m)*hypothesis(theta_0,theta_1,x[j],y[j])*x[j]
        
        # Append updated theta0 and theta1
        theta0.append(theta_0)
        theta1.append(theta_1)
        
        # Append cost function value        
        J.append((1/2*m)*(hypothesis(theta_0,theta_1,x[j],y[j]))**2)
           
    # Find minimized cost function value
    J0 = min(J, key=abs)
    
    # Get corresponding Thetas for minimized cost function
    flag = -1
    for i in J:
        flag = flag + 1
        if i == J0:
            theta_0_opt = theta0[flag]
            theta_1_opt = theta1[flag]
    
    optimized_thetas = [theta_0_opt, theta_1_opt] 
    
    return optimized_thetas
