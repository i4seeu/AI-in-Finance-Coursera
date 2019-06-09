import os
import numpy as np 
import math as m 

import matplotlib as plt

from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 

import tensorflow as tf

def reset_graph(seed = 42):
    #to make the results reproducible across runs
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
#generate data
n_points = 5000
n_features = 3

bias = np.ones(n_points).reshape(-1,1)
low = np.ones((n_points,n_features),'float')
high = np.ones((n_points,n_features),'float')

#simulated features
X = np.random.uniform(low=low,high=high)
#print(X.shape)
#simulated noises 
noise = np.random.normal(size=(n_points,1))

#outputs
weights = np.array([1.0,0.5,0.2,0.1])
noise_std = 0.1
Y = weights[0] * bias + np.dot(X,weights[1:]).reshape((-1,1)) + noise_std * noise
#print(Y.shape)
#split to the train and test set
train_test_split = 4 #1/4 of the data is used for a test
n_test = int(n_points/train_test_split)
n_train = n_points - n_test

X_train = X[:n_train,:]
Y_train = Y[:n_train].reshape((-1,1))

X_test = X[n_train:,:]
Y_test = Y[n_train:].reshape((-1,1))

#Linear regression with sklearn
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
print(np.r_[lin_reg.intercept_.reshape(-1,1),lin_reg.coef_.T])
