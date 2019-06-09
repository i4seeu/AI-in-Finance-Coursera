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
#print(X)
