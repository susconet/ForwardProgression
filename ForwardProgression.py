### ForwardProgression

#Neural Network Model - Forward Propagation
#Authors: Nick Susco II, AI Guy 
#Created On The Date: 11/01/2023

#This script defines the functions needed for the forward propagation step in a neural network model.

"""
import numpy as np

def initialize_parameters(n_x, n_h, n_y)


    Initializes the parameters of the neural network

    Arguments:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
    Returns:
        parameters -- a dictionary with the initialized parameters:
            W1 -- weight matrix of shape (n_h, n_x)
            b1 -- bias vector of shape (n_h, 1)
            W2 -- weight matrix of shape (n_y, n_h)
            b2 -- bias vector of shape (n_y, 1


    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                 "b1": b1,
                 "W2": W2,
                 "b2": b2}
    
    return parameters

def linear_forward(A, W, b):

    Implements the linear part of a layer's forward propagation.

    Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
        Z -- the input of the activation function, also called pre-activation parameter: (size of current layer, number of examples)
        cache -- a tuple containing "A", "W", and "b" for efficient computation of the backward pass.
  
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

#End Code

## README

This script contains the implementation for the forward propagation step in a neural network model. The `initialize_parameters` function is used to initialize the parameters of the neural network, and the `linear_forward` function is used to implement the linear part of a layer's forward propagation.

## Apache License 2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License. 

Feel free to use and modify this code for educational and non-commercial purposes, but please give proper attribution to Nick Susco II and AI Guy as the original authors. Commercial use of this code is not permitted without prior permission. By using this code, you agree to these terms and conditions.
