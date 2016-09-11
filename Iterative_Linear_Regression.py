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

# Define training elements and learning rate
m = 10
alpha = 0.4

def cost_function(t0,t1,x,y):
    return t0 + t1*x - y

def gradient_decent(x,y):
    
    # Initial Conditions
    theta_0 = 0
    theta_1 = 0
    theta0 = []
    theta1 = []
    J = []
    
    # Iterate over range m to calculate theta1 and theta0
    for j in range(m):
        
        # calculate the gradient decent
        theta_0 = theta_0 - alpha*(1/m)*cost_function(theta_0,theta_1,x[j],y[j])
        theta_1 = theta_1 - alpha*(1/m)*cost_function(theta_0,theta_1,x[j],y[j])*x[j]
        
        # Append updated theta0 and theta1
        theta0.append(theta_0)
        theta1.append(theta_1)
        
        # Append cost function value        
        J.append((1/2*m)*(theta_0 + theta_1*x[j] - y[j]))
           
    return theta0, theta1
    
# Generate random data points to optimize    
x = []
y = []
for i in range(m):
    x.append(i)
    y.append(random.uniform(0,50))
    
   
gradient_decent(x,y)

# Find minimized cost function value
J0 = min(J, key=abs)

# Get corresponding Thetas for minimized cost function
flag = -1
for i in J:
    flag = flag + 1
    if i == J0:
        theta_0_opt = theta0[flag]
        theta_1_opt = theta1[flag]

new_y = []      
for k in range(m):
    new_y.append(theta_0_opt + theta_1_opt*k)

plt.plot(x,new_y)
plt.scatter(x,y)
plt.plot(x,J)
plt.show()


