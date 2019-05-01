# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:30:03 2019

@author: PIANDT
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image


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

#This is a regularization parameter meant to randomly select and drop of some nerons
#Like any regularization term, it helps prevent overfitting
regParameter = 0.5 

#TensorFlow Graph design
#the neural network will be setup as computational graph

#TensorFlow uses tensors, data structures similar to lists/arrays
# The tensors are initialized, processed and updated as they are passed through the graph during learning
inputTensor = tf.placeholder("float", [None, inputLayer])   #input tensor. None==infinite training examples
outputTensor = tf.placeholder("float", [None, outputLayer])   #output tensor for possible value; 0-9
regTensor = tf.placeholder(tf.float32)  #tensor for the regularization parameter

# weight and bias will be constantly updated throuout the training process
# they are representations of the relationship between units -- also used in Activation Functions

#Weight Tensor
weightsTensor = {
    'weight1': tf.Variable(tf.truncated_normal([inputLayer, firstHiddenLayer], stddev=0.1)),
    'weight2': tf.Variable(tf.truncated_normal([firstHiddenLayer, secondHiddenLayer], stddev=0.1)),
    'weight3': tf.Variable(tf.truncated_normal([secondHiddenLayer, thirdHiddenLayer], stddev=0.1)),
    'outLayerWeight': tf.Variable(tf.truncated_normal([thirdHiddenLayer, outputLayer], stddev=0.1)),
}

#Bias Tensor
biasTensor = {
    'bias1': tf.Variable(tf.constant(0.1, shape=[firstHiddenLayer])),
    'bias2': tf.Variable(tf.constant(0.1, shape=[secondHiddenLayer])),
    'bias3': tf.Variable(tf.constant(0.1, shape=[thirdHiddenLayer])),
    'outLayerBias': tf.Variable(tf.constant(0.1, shape=[outputLayer]))
}

#For each layer, we calculate matrix multiplication on output of its preceding layer
# We consider current layer’s weight and add the bias
# In the last hidden layer (The layer before the output layer), we apply a regularization(dropout) of 0.5

layer1WB = tf.add(tf.matmul(inputTensor, weightsTensor['weight1']), biasTensor['bias1'])
layer2WB = tf.add(tf.matmul(layer1WB, weightsTensor['weight2']), biasTensor['bias2'])
layer3WB = tf.add(tf.matmul(layer2WB, weightsTensor['weight3']), biasTensor['bias3'])
layerDrop = tf.nn.dropout(layer3WB, regParameter)
outputLayerWB = tf.matmul(layer3WB, weightsTensor['outLayerWeight']) + biasTensor['outLayerBias']

# Loss function definition. A perfect loss function outputs 0. We will use TF log loss or cross-entropy in this network.
#gradient descent optimization: we will use rthe Adam optimizer 
#Takes iterative steps along the gradient in a negative (descending) direction
#With expectation to find the local or global minimum of a function

crossEntropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=outputTensor, logits=outputLayerWB
        ))
trainingStep = tf.train.AdamOptimizer(1e-4).minimize(crossEntropy)

#During training, datasets are passed through the graph and loss function is optimized per iteration. 
#As the neural network iterates through several training sets, the loss is reduced to ensure a more accurate prediction.
# Testing involves running our test datasets through the graph trained by our trainingData 
#and comparing if the predictions are accurate or not.

accuratePrediction = tf.equal(tf.argmax(outputLayerWB, 1), tf.argmax(outputTensor, 1))
predictionAccuracy = tf.reduce_mean(tf.cast(accuratePrediction, tf.float32))

#arg_max compares accurately predicted images by comparing 
#predictions(OutputLayerWB) to actual labels(Output tensor)
# equal function returns accurate predictions as Boolean list. 
#List is floated and mean is calculated as the total accuracy

#initialization of a graph session
initGraph = tf.global_variables_initializer()
graphSession = tf.Session()
graphSession.run(initGraph)

#Neural network training is meant to optimize the cost function as much as possible. 
#i.e reducing the gap between the predicted and actual value. 
#Four main steps involved are;
#Forward propagation, Calculate cost/loss, Back Propagation and Update parameters to reduce loss for next step

# training on batches
for i in range(numberOfIterations):
    inputBatch, outputBatch = mnist.train.next_batch(batchSize)
    graphSession.run(trainingStep, feed_dict={
        inputTensor: inputBatch, outputTensor: outputBatch, regTensor: regParameter
        })

    # print loss and accuracy for each batch trained
    if i % 100 == 0:
        trainBatchLoss, trainBatchAccuracy = graphSession.run(
            [crossEntropy, predictionAccuracy],
            feed_dict={inputTensor: inputBatch, outputTensor: outputBatch, regTensor: 1.0}
            )
        print(
            "Iteration",
            str(i),
            "\t| Loss =",
            str(trainBatchLoss),
            "\t| Model Accuracy =",
            str(trainBatchAccuracy)
            )

# After 150 iterations of each training step, we provide a training set to the neural network.
# Assess the model accuracy and loss
#accuracy and loss per batch may not necessarily decrease as expected for whole model
# reason for training in batches is to reduce computational cost
# and to allow the network explore different examples before optimization

#TESTING THE MODEL
# after training, model is assessed using the testDataset with a dropout of 1.0 to ensure all units are active for testing
testDataAccuracy = graphSession.run(predictionAccuracy, feed_dict={inputTensor: mnist.test.images, outputTensor: mnist.test.labels, regTensor: 1.0})
print("\nTest Data Accuracy  :", testDataAccuracy)


#Improving Accuracy -- Alterable parameters
#learning rate, regParameter, hidden layer units, batch size, no. of hidden layers and iterations.

testImage = np.invert(Image.open("handwrittenDigitImage2.png").convert('L')).ravel()

#now we can predict what handwritten digit is on this image
predictHandDigit = graphSession.run(tf.argmax(outputLayerWB, 1), feed_dict={inputTensor: [testImage]})
print ("Written digit in image is :", np.squeeze(predictHandDigit))




