from __future__ import division
"""
Copyright Derek Black 2016

Required Modules:
Numpy
Itertools

Topics include:
Gradient Decent
Linear Regression
Cost Functions
Normal Equations

This module is for multi-varient problems.
It utilzies two methods: Gradient Decent to Optimize the cost function
and normal equations to automatically find the optimized thetas for 
a minimized cost function.

How to use iLinearRegression:
iLinearRegression is an iterative method for solving linear regression problems.
The i in LinearRegression stands for iterative.
iLinearRegression utilizes a method called Gradient Decent to find values of theta
for a minimized cost function. The function returns the optimized values of theta.
This function is for single varient systems.
	
Ex. [theta0,theta1] = iLinearRegression([1,2,3],[10,12,15],m=10,alpha=0.4)
The returned values can now be used in the equation 'y= theta0 + theat1*x' for
the appropriate best fit linear model.

LinearRegression is a normal equation based linear regression method.
The user inputs his training sets of size m, as well as the out y.
The function will return a list of optimized thetas.

"""

def hypothesis(t0,t1,x,y):
    return (t0 + t1*x - y)

# Adjust training elements (m) and learning rate (alpha) according to data
def iLinearRegression(x,y,m=10,alpha=0.4):
	
	# *************************************************************************
	# Initial Conditions
	# *************************************************************************
    theta_0 = 0
    theta_1 = 0
    theta0 = []
    theta1 = []
    J = []
    
				
	# *************************************************************************
	# Gradient Decent
	# *************************************************************************
    for j in range(m): # Iterate over range m to calculate theta1 and theta0
        
        # calculate the gradient decent
        theta_0 = theta_0 - alpha*(1/m)*hypothesis(theta_0,theta_1,x[j],y[j])
        theta_1 = theta_1 - alpha*(1/m)*hypothesis(theta_0,theta_1,x[j],y[j])*x[j]
        
        # Append updated theta0 and theta1
        theta0.append(theta_0)
        theta1.append(theta_1)
        
        # Append cost function value        
        J.append((1/2*m)*(hypothesis(theta_0,theta_1,x[j],y[j]))**2)
           
    
    J0 = min(J, key=abs)  # Find minimized cost function value
    
    # Get corresponding Thetas for minimized cost function
    flag = -1
    for i in J:
        flag = flag + 1
        if i == J0:
            theta_0_opt = theta0[flag]
            theta_1_opt = theta1[flag]
    
    optimized_thetas = [theta_0_opt, theta_1_opt] 
    
    return optimized_thetas
				
def LinearRegression(*argv):
	import numpy as np
	from numpy.linalg import inv
	from itertools import chain
	
	"""
	To use LinearRegression, enter your training data x, and output y (type: list)
	ex. LinearRegression(x1,x2,x3,x4........,xn,y)
		where x and y = [data]

	"""
	
	# Initialize Variables
	ones = []
	popmat = []
	
	# Generate ones vector for x0, training setting data column 1
	for i in range(len(argv[0])):
		ones.append(1)
		
	popmat.append(ones)
	
	# *************************************************************************
	# Gather features and output from user to matrix
	# *************************************************************************
	i = -1
	for arg in argv:
		i += 1
		popmat.append(arg)
		if i == (len(argv) - 2):
			break
		
	# Convert x and y matricies to proper dimenions
	y = np.transpose(np.matrix(argv[-1]))		
	x = np.transpose(np.matrix(popmat))
	
	# Calculate the optimal thetas
	thetas = inv(np.transpose(x)*x)*np.transpose(x)*y
	
	convth = thetas.tolist()
	optimized_thetas = list(chain.from_iterable(convth))

	
	return optimized_thetas
