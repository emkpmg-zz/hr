# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:30:03 2019

@author: PIANDT
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Use one-hot encoding for output labels
#with all digits ranging from 0-9
# 3 == vector [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# value at index 3 == 1
#Import/Explore data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#exploring size of the dataset for testing and training
trainingData = mnist.train.num_examples
validationData = mnist.validation.num_examples
testData = mnist.test.num_examples

#Neural network design
# The input layer will include all the unrolled pixels (28x28 pixels) for each image
inputLayer = 784  

firstHiddenLayer = 512
secondHiddenLayer = 256
thirdHiddenLayer = 128

# Our neural network should predict a handwritten digit of numbers 0 - 9
outputLayer = 10

#a deep neural network, 
#i.e a neural network with too many hidden layers is computationally expensive
#Though can be more efficient and accurate in prediciting too

#hyperparameters in neural netowrks are continuously updated while training
# They can be given initial values and remain constant throughout the process.

learningRate = 1e-4 #step size of gradient descent in the move towards global minimum
numberOfIterations = 1000 #How many time or steps is enough to reach the global minimum
batchSize = 128 #can be reduced to 64 or 16 if you running out of memory
dropout = 0.5 #