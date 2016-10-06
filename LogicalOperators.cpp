#include "stdafx.h"
#include "math.h"
#include "windows.h"

//*****************************************************************************************
// Copyright (c) Derek Black 2016
// This is a simple program to compute logical 'and' and 'or' based on passed values
// The application uses a simple neural network with one hidden layer and two external layers
//*****************************************************************************************

//*****************************************************************************************
// Internal Function Prototypes
//*****************************************************************************************
int _and(float x1, float x2); // Set up logical 'and' function
int _or(float x1, float x2); // Set up logical 'or' function
float sigmoid(float x); // Set up sigmoid function

void main()
{
	// Test logic functions
	// Set up x1 and x2
	int x1 = 0;
	int x2 = 1;

	// Compute the logical operators
	bool a = _and(x1, x2);
	bool b = _or(x1, x2);

	printf("and = %d, or = %d ", a, b);
	system("pause");

    return;
}

// Sigmoid function to determine state of logical operator passed
float sigmoid(float x)
{
	bool state;
	float sig = 1 / (1 + exp(-x)); // Compute the sigmoid function of passed x

	// Conditionals to determine x state
	if (sig >= 0.5) { state = true; }
	else if (sig < 0.5) { state = false; };

	return state;

}

int _and(float x1, float x2) 
{
	// Set up corresponding weights for 'and' function
	int weight[3] = { -30, 20, 20 };

	// Compute corresponding total weight
	float compx = 1 * weight[0] + x1 * weight[1] + x2 * weight[2];

	bool truth = sigmoid(compx); // Get state of sigmoid function based on passed x1 x2

	return truth;
}

int _or(float x1, float x2)
{
	// Set up corresponding weights for 'or' function
	int weight[3] = { -10, 20, 20 };

	// Compute corresponding total weight
	float compx = 1 * weight[0] + x1 * weight[1] + x2 * weight[2];

	bool truth = sigmoid(compx); // Get state of sigmoid function based on passed x1 x2

	return truth;
}
