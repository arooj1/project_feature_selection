# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:17:33 2021

@author: quresa9
"""

'''
Purpose¶
Feature Selection by optimizing FSJaya Algorithm.
The challenge is to tweak the present algorithm for feature selection purposes.
'''

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#import os
#from project_part_2 import sigmoid_function, random_r
import project_part_1 as pj1
import classifiers as cl
import test_cases as t

# Step 1: Load datasets

madelon = pd.read_csv('madelon/madelon_train.data', sep=' ', header = None)
madelon.info()
madelon.head()

madelon_test = pd.read_csv('madelon/madelon_test.data', sep=' ', header = None)
madelon_test.info()
#madelon_test.head()

musk =  pd.read_csv('musk/clean2.data/clean2.data', header = None)
musk.info()
musk.head()

musk_test =  pd.read_csv('musk/clean1.data/clean1.data', header = None)
#musk_test
'''
Step 2: Prepare data for Classifiers
According to the shared paper [A jaya algorithm based wrapper method for optimal feature selection in supervised classification] data has to be trained using following classifiers:

- NB (Naive Bayes)
- KNN (K Nearest Neighbor)
- LDA 
- RT (Regression Tree)

'''
Musk_X = musk.drop(168, axis = 1)
Musk_X = Musk_X.drop(0, axis=1)
Musk_X = Musk_X.drop(1, axis=1)
Musk_X.columns = np.arange(len(Musk_X.columns))
Musk_Y = musk[168]


Madelon_X = madelon.drop(500, axis = 1)
Y_labels = np.hstack([np.ones(1000), np.zeros(1000)])
Madelon_Y = pd.Series(Y_labels.T)

# Step 2b : Separating features and labels as X, Y variables  (Test Dataset)
Musk_x = musk_test.drop(168, axis = 1)
Musk_x = Musk_x.drop(0, axis=1)
Musk_x = Musk_x.drop(1, axis=1)
Musk_x.columns = np.arange(len(Musk_x.columns))
Musk_y = musk_test[168]


Madelon_x = madelon_test.drop(500, axis = 1)
y_labels = np.hstack([np.ones(900), np.zeros(900)])
Madelon_y = pd.Series(y_labels.T)
len(Musk_x)
len(Musk_y)
print('***********************************************************************************')
print('************************************ DATASETS *************************************')
print('***********************************************************************************')
print("Madelon TRAINING \n dataset: # of features: " , Madelon_X.shape[1], 
      '\n Number of Measurements: ', Madelon_X.shape[0], 
      'Y_shape: ', Madelon_Y.shape)

print("Madelon TEST \n dataset: # of features: " , Madelon_x.shape[1], 
      '\n Number of Measurements: ', Madelon_x.shape[0], 
      'Y_shape: ', Madelon_y.shape)

print("Musk TRAINING \n dataset: # of features: " , Musk_X.shape[1], 
      '\n Number of Measurements: ', Musk_X.shape[0],
      'Y_shape: ', Musk_Y.shape)

print("Musk TEST \n dataset: # of features: " , Musk_x.shape[1], 
      '\n Number of Measurements: ', Musk_x.shape[0], 
      'Y_shape: ', Musk_y.shape)

'''
Step 3: Classification¶
Step 3a: Train Classifiers without Feature Selection
'''
print('***********************************************************************************')
print('******************* BEFORE JAYA ALGORITHM FEATURE SELECTION ***********************')
print('***********************************************************************************')

print('\n\n ----------------------- MEDELON DATASET ---------------------')
madelon_return = cl.data_classification(train_x = Madelon_X, train_y = Madelon_Y, test_x = Madelon_x ,test_y = Madelon_y)
print('\n\n ----------------------- MUSK DATASET ---------------------')
musk_return = cl.data_classification(train_x = Musk_X, train_y = Musk_Y, test_x = Musk_x ,test_y = Musk_y)

# Step 3b: PROPOSED STRATEGY with Feature Selection
t.test_cases(Musk_X, Musk_Y, Musk_x, Musk_y, Madelon_X, Madelon_Y, Madelon_x, Madelon_y)