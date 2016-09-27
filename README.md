# MACHINE LEARNING

###SUPERVISED LEARNING TOPICS

#####Iterative Methods for Gradient Decent

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

theta = iLinearRegression(x,y,1500,0.4)

######Where x is the input set, y output set. The 1500 is the number of iterations to perform, and alpha is the learning rate.
######Alpha should be set according to your data. It is important to select the right learning rate so that gradient decent actually converges. Too large a value and the algorithm could miss the optimal value, to little and the algorithm can take to long too converge.

Which fits the form of the linear equation 'y = theta_0 + theata_1*x'

It is recommended that the user uses plotting tools like matplotlib to visualize the regression.


###Ex. 2, Normal Function Linear Regression using LinearRegression()

####Minimize the cost function using the normal equation for the data set...

input(x) = [1,2,3,4,5], output(y) = [10,0,20,40,-20]

######This algorithm is great for multiple training sets, and accepts m sets. It should be noted that while this algorithm does converge to the optimal solution for theta, computation time may be effected by to large a set. If the users training set is larger than m=10,000-100,000 it might be recommended to use an iterative gradient decent method (stil in development for m>1 data sets).

Solution:

[theta_0, theta_1] = LinearRegression(x,y)

Which fits the form of the linear equation 'y = theta_0 + theata_1*x'

It is recommended that the user uses plotting tools like matplotlib to visualize the regression.
