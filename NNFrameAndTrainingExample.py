# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:24:36 2024

@author: anton
"""
import numpy as np
#import csv
#import pandas as pd
#import os
import torch
import matplotlib.pyplot as plt
#%% nn model with 2 hidden layers
#Defining Neural Network Model
class NN(torch.nn.Module):
    def __init__(self,n_feature,n_hidden1,n_hidden2,n_output):
        super(NN,self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature,n_hidden1)
        #self.do1 = torch.nn.Dropout(0.15)
        #self.relu1 = torch.nn.LeakyReLU()
        #self.bn1 = torch.nn.BatchNorm1d(n_hidden1,affine=False)
        self.hidden2 = torch.nn.Linear(n_hidden1,n_hidden2)
        #self.bn2 = torch.nn.BatchNorm1d(n_hidden2,affine=False)
        #self.relu2 = torch.nn.LeakyReLU()
        #self.do2 = torch.nn.Dropout(0.1)
        self.predict = torch.nn.Linear(n_hidden2,n_output)
        
        
    def forward(self,x):
        x = self.hidden1(x)
        x = torch.sigmoid(x)
        #x = self.do1(x)
        x = self.hidden2(x)
        x = torch.sigmoid(x)
        #x = self.do2(x)
        x = self.predict(x)
        return x
#%% instanse of NN model
#instantiate the Neural Network
#model = NN(n_feature=2,n_hidden1=17,n_hidden2=7, n_output=1)
model = NN(n_feature=2,n_hidden1=17,n_hidden2=31, n_output=1)

#Define loss function : 
# here we use Mean Square Error as the loss function
loss_func = torch.nn.MSELoss()

#Define the optimizer that should be used in training the Neural Network.
# Here 'lr' is the learning rate
optimizer = torch.optim.Adam(model.parameters(),lr=0.1)

#%% get a training set and a test set of data


#%% training model
#TODO: train the Neural network model by changing the hyper parameters such as learning rate, number of epochs, number of neurons in hidden layers of the neural network.
# What is the minimum mean square error that you can achieve as your neural network converges for the training set.
#  (you will be able to achive a MSE of less than 10 as the Neural network converges.)
num_epochs = 2200
losslist = []
for _ in range(num_epochs):
    prediction = model(train_set_inputs) # Forward pass prediction. Saves intermediary values required for backwards pass
    loss = loss_func(prediction, train_set_targets) # Computes the loss for each example, using the loss function defined above
    optimizer.zero_grad() # Clears gradients from previous iteration
    loss.backward() # Backpropagation of errors through the network
    optimizer.step() # Updating weights
    print("prediction =",prediction)
    print("Loss: ", loss.detach().numpy())
    losslist.append(loss.detach().numpy())

#plot the mean square error in each epoch/iteration
plt.plot(np.arange(len(losslist)),losslist)
plt.show()

#%% test the nn model on the test set

#TODO: feed the normalized test set inputs to the Neural Network model and obtain the prediction for the test set.
prediction_test = model(test_set_inputs)

print(prediction_test.shape)

#plot the prediction error of the test set
test_set_prediction_error = prediction_test - test_set_targets  #should probably do it as mse

plt.plot(np.arange(len(test_set_prediction_error.tolist())),test_set_prediction_error.tolist())

