# MACHINE LEARNING

###SUPERVISED LEARNING TOPICS

#####Iterative methods for gradient decent

#####Cost Functions

#####Normal Equations

#####Logistic Regression

###UNSUPERVISED LEARNING TOPICS:
#####Coming soon.

This project is frequently updated.

## PURPOSE OF THIS LIBRARY

The main purpose of this library is to provide users with an alternative approach to machine learning. There are many good machine learning libraries that exist in python, but many of them are not useful in providing useful feedback and usability to users who are new to machine learning. The aim of this project is to simplify the approach to machine learning problems, with easy input easy output functions. This library is not for advanced users wishing for more robust control over their data. This library targets begining users who wish to learn more about machine learning. 

## HOW TO USE


###Ex. 1, Iterative Linear Regression using iLinearRegression()

####Minimize the cost function iteratively for the data set ...

input(x) = [1,2,3,4,5], output(y) = [10,0,20,40,-20]

Solution:

[theta_0, theta_1] = iLinearRegression(x,y,m=5,alpha=0.4)

Which fits the form of the linear equation 'y = theta_0 + theata_1*x'


###Ex. 2, Normal Function Linear Regression using LinearRegression()

####Minimize the cost function using the normal equation for the data set...

input(x) = [1,2,3,4,5], output(y) = [10,0,20,40,-20]

Solution:

[theta_0, theta_1] = LinearRegression(x,y)

Which fits the form of the linear equation 'y = theta_0 + theata_1*x'
