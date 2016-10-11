"""
Derek Black

This is a simple 3 layer neural network. It can take as many inputs/outputs as
you give it. It is very simple to train your data with this module. It is important
that you input your data as numpy matricies otherwise error codes will be generated.

Typical formatting for data input:


    | --x1^T-- |      | y1 |
X = | --x2^T-- |  y = | y2 |
    | --x3^T-- |      | y3 |
    
By default automatic data formatting is turned on when passing data to the 
training function. If you feel your data is not being formatted appropriately,
enter form = 'off'.

Simple Ex.
    x = np.matrix([[0,0],[0,1],[1,0],[1,1]]) 
    y = np.matrix([[0,0,1,1]]) 
    l = np.matrix([1,0])
    [th1,th2] = train(x,y)
    prediction = predict(l,th1,th2)
    print prediction

"""

import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
    
def formatdata(x,y):
 
    if isinstance(x,(np.ndarray, np.generic)) and isinstance(y,(np.ndarray, np.generic)) == True:
        [m_in,n_in] = map(int,x.shape) # Get dimensions of training set
        [m_out,n_out] = map(int,y.shape) # Get dimensions of output
    else:
        raise ValueError('Input/Output data must be of type Numpy Array/Matrix')
    
    # Determine if x needs to be tranposed
    if m_in > n_in:
        x = x
    elif n_in > m_in:
        x = x.T
    else:
        x = x
    
    # Determine if y needs to be tranposed
    if m_out > n_out:
        y = y
    elif n_out > m_out:
        y = y.T
    else:
        y = y
    
    return x,y

def train(x,y,nhid=3,iterations=60000,form = 'on'):
    if form == 'on':
        [x,y] = formatdata(x,y)
        print "Data has been formatted:\n"
        print "X = \n"
        print x
        print "\n"
        print "Y = \n"
        print y
        print "\n"
    elif form == 'off':
        pass
    
    [m_in,n_in] = map(int,x.shape) # Get dimensions of training set
    [m_out,n_out] = map(int,y.shape) # Get dimensions of output

    # Set up random weight matricies
    W0 = 2*np.random.random(( n_in + 1, nhid + 1 )) - 1
    W1 = 2*np.random.random(( nhid + 1 , n_out )) - 1
    
    # Initialize layer 0
    layer0 = np.concatenate((x,np.ones((m_in,1))),axis=1)
    
    for i in xrange(iterations):
        
        # Forward Prop.
        layer1 = sigmoid(np.dot(layer0,W0))
        layer2 = sigmoid(np.dot(layer1,W1))
        
        
        # Backwards Prop.
        layer2_error = y - layer2
        layer2_del = np.multiply(layer2_error,np.multiply(layer2,(1-layer2)))
        
        layer1_error = layer2_del.dot(W1.T)
        layer1_del = np.multiply(layer1_error,np.multiply(layer1,(1-layer1)))
        
        if i%10000 == 0:
            print "error layer 1:" + '    ' + str(np.abs(np.mean(layer1_error)))
            print "error layer 2:" + '    ' + str(np.abs(np.mean(layer2_error))) + '\n'
            
        # Update weight matricies
        W1 = W1 + layer1.T.dot(layer2_del)
        W0 = W0 + layer0.T.dot(layer1_del)
        
    return W0,W1
    
    
def predict(x,W0,W1):
    
    [m_in,n_in] = map(int,x.shape) # Get dimensions of training set
    
    # Feed Forward through the network to make a prediction
    layer0 = np.concatenate((x,np.ones((m_in,1))),axis=1)
    layer1 = sigmoid(np.dot(layer0,W0))
    
    prediction = sigmoid(np.dot(layer1,W1))
        
    prediction = np.array(prediction).reshape(-1,).tolist()
    
    output = []
    for p in prediction:
        if p >= 0.5:
            output.append(1)
        else:
            output.append(0)
    
    return output


def demo():
    x = np.matrix([[0,0],[0,1],[1,0],[1,1]]) 
    y = np.matrix([[0,0,1,1]]) 
    l = np.matrix([1,0])
    [th1,th2] = train(x,y)
    prediction = predict(l,th1,th2)
    print prediction
    
demo()