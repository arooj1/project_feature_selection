# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 20:53:42 2021

@author: quresa9

TEST USE CASE 
 
The purpose of this file is to generate maximum test cases using different combinations of parameters. 

"""


import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#import os
#from project_part_2 import sigmoid_function, random_r
import project_part_1 as pj1
import classifiers as cl

def test_cases(Musk_X, Musk_Y, Musk_x, Musk_y, Madelon_X, Madelon_Y, Madelon_x, Madelon_y): 
    '''
    obj_function = 'AUC'
    pop = 10
    sigmoid_function = 'default'
    bin_function = 'random_r'  
    print('********************************** AFTER ***********************************')
    print('TEST_CASE - 1')
    print('---------------------  PARAMETERS ------------------------')
    print(f'\n Population: {pop},\n Objective Function: {obj_function},\n Binary Conversion Function: {bin_function},\n Sigmoid Function: {sigmoid_function}')
    print('********************************** AFTER ***********************************')
    for a in range(3):
        print('********************************** AFTER ***********************************')
        print(' --------------------- MUSK DATASET RESULT # --------------------------- ', a)
        print('****************************************************************************')
            
        dsp = pj1.feature_optimization(trainX = Musk_X,
                                   trainY = Musk_Y,
                                   testX = Musk_x,
                                   testY = Musk_y, 
                                   population =pop ,
                                   obj_function= obj_function,
                                   s_function = sigmoid_function,
                                   binary_function = bin_function)
    
        MuskX_OP, Muskx_OP = dsp()
        print('Number of Features ', MuskX_OP.shape[1])
        
        musk_return = cl.data_classification(train_x = MuskX_OP, train_y = Musk_Y, 
                                      test_x = Muskx_OP ,test_y = Musk_y)
        
        #pd.DataFrame(musk_return).to_json(str(a)+str('_Musk_t1_')+str(obj_function)+'_'+str(sigmoid_function)+'_'+str(bin_function)+str('.csv'), index=False)
        
    for a in range(3):
        print('********************************** AFTER ***********************************')
        print(' --------------------- MADELON DATASET RESULT # --------------------------- ', a)
        print('****************************************************************************')
            
        dsp = pj1.feature_optimization(trainX = Madelon_X,
                                   trainY = Madelon_Y,
                                   testX = Madelon_x,
                                   testY = Madelon_y, 
                                   population =pop ,
                                   obj_function= obj_function,
                                   s_function = sigmoid_function,
                                   binary_function = bin_function)
    
        MadelonX_OP, Madelonx_OP = dsp()
        print('Number of Features ', MadelonX_OP.shape[1])
        
        madelon_return = cl.data_classification(train_x = MadelonX_OP, train_y = Madelon_Y, 
                                      test_x = Madelonx_OP ,test_y = Madelon_y)
        
        
        #pd.DataFrame(madelon_return).to_json(str(a)+str('_Madelon_t1_')+str(obj_function)+'_'+str(sigmoid_function)+'_'+str(bin_function)+str('.csv'), index=False)
    
    obj_function = 'ER'
    pop = 10
    sigmoid_function = 'default'
    bin_function = 'random_r'  
    print('********************************** AFTER ***********************************')
    print('TEST_CASE - 2')
    print('---------------------  PARAMETERS ------------------------')
    print(f'\n Population: {pop},\n Objective Function: {obj_function},\n Binary Conversion Function: {bin_function},\n Sigmoid Function: {sigmoid_function}')
    print('********************************** AFTER ***********************************')
    for a in range(3):
        print('********************************** AFTER ***********************************')
        print(' --------------------- MUSK DATASET RESULT # --------------------------- ', a)
        print('****************************************************************************')
            
        dsp = pj1.feature_optimization(trainX = Musk_X,
                                   trainY = Musk_Y,
                                   testX = Musk_x,
                                   testY = Musk_y, 
                                   population =pop ,
                                   obj_function= obj_function,
                                   s_function = sigmoid_function,
                                   binary_function = bin_function)
    
        MuskX_OP, Muskx_OP = dsp()
        print('Number of Features ', MuskX_OP.shape[1])
        
        musk_return = cl.data_classification(train_x = MuskX_OP, train_y = Musk_Y, 
                                      test_x = Muskx_OP ,test_y = Musk_y)
        #pd.DataFrame(musk_return).to_json(str(a)+str('_Musk_t2_')+str(obj_function)+'_'+str(sigmoid_function)+'_'+str(bin_function)+str('.csv'), index=False)    
        
    for a in range(3):
        print('********************************** AFTER ***********************************')
        print(' --------------------- MADELON DATASET RESULT # --------------------------- ', a)
        print('****************************************************************************')
            
        dsp = pj1.feature_optimization(trainX = Madelon_X,
                                   trainY = Madelon_Y,
                                   testX = Madelon_x,
                                   testY = Madelon_y, 
                                   population =pop ,
                                   obj_function= obj_function,
                                   s_function = sigmoid_function,
                                   binary_function = bin_function)
    
        MadelonX_OP, Madelonx_OP = dsp()
        print('Number of Features ', MadelonX_OP.shape[1])
        
        madelon_return = cl.data_classification(train_x = MadelonX_OP, train_y = Madelon_Y, 
                                      test_x = Madelonx_OP ,test_y = Madelon_y) 
        
        
        #pd.DataFrame(madelon_return).to_json(str(a)+str('_Madelon_t2_')+str(obj_function)+'_'+str(sigmoid_function)+'_'+str(bin_function)+str('.csv'), index=False)
        
    obj_function = 'AUC'
    pop = 10
    sigmoid_function = 'arctangent_function'
    bin_function = 'random_r'  
    print('********************************** AFTER ***********************************')
    print('TEST_CASE - 3')
    print('---------------------  PARAMETERS ------------------------')
    print(f'\n Population: {pop},\n Objective Function: {obj_function},\n Binary Conversion Function: {bin_function},\n Sigmoid Function: {sigmoid_function}')
    print('********************************** AFTER ***********************************')
    for a in range(2):
        print('********************************** AFTER ***********************************')
        print(' --------------------- MUSK DATASET RESULT # --------------------------- ', a)
        print('****************************************************************************')
            
        dsp = pj1.feature_optimization(trainX = Musk_X,
                                   trainY = Musk_Y,
                                   testX = Musk_x,
                                   testY = Musk_y, 
                                   population =pop ,
                                   obj_function= obj_function,
                                   s_function = sigmoid_function,
                                   binary_function = bin_function)
    
        MuskX_OP, Muskx_OP = dsp()
        print('Number of Features ', MuskX_OP.shape[1])
        
        musk_return = cl.data_classification(train_x = MuskX_OP, train_y = Musk_Y, 
                                      test_x = Muskx_OP ,test_y = Musk_y)
        
        #pd.DataFrame(musk_return).to_json(str(a)+str('_Musk_t3_')+str(obj_function)+'_'+str(sigmoid_function)+'_'+str(bin_function)+str('.csv'), index=False)
        
    for a in range(2):
        print('********************************** AFTER ***********************************')
        print(' --------------------- MADELON DATASET RESULT # --------------------------- ', a)
        print('****************************************************************************')
            
        dsp = pj1.feature_optimization(trainX = Madelon_X,
                                   trainY = Madelon_Y,
                                   testX = Madelon_x,
                                   testY = Madelon_y, 
                                   population =pop ,
                                   obj_function= obj_function,
                                   s_function = sigmoid_function,
                                   binary_function = bin_function)
    
        MadelonX_OP, Madelonx_OP = dsp()
        print('Number of Features ', MadelonX_OP.shape[1])
        
        madelon_return = cl.data_classification(train_x = MadelonX_OP, train_y = Madelon_Y, 
                                      test_x = Madelonx_OP ,test_y = Madelon_y) 
        
        #pd.DataFrame(madelon_return).to_json(str(a)+str('_Madelon_t3_')+str(obj_function)+'_'+str(sigmoid_function)+'_'+str(bin_function)+str('.csv'), index=False)
    '''
    obj_function = 'AUC'
    pop = 10
    sigmoid_function = 'hyperbolic_function'
    bin_function = 'default_r'  
    print('********************************** AFTER ***********************************')
    print('TEST_CASE - 4')
    print('---------------------  PARAMETERS ------------------------')
    print(f'\n Population: {pop},\n Objective Function: {obj_function},\n Binary Conversion Function: {bin_function},\n Sigmoid Function: {sigmoid_function}')
    print('********************************** AFTER ***********************************')
    for a in range(2):
        print('********************************** AFTER ***********************************')
        print(' --------------------- MUSK DATASET RESULT # --------------------------- ', a)
        print('****************************************************************************')
            
        dsp = pj1.feature_optimization(trainX = Musk_X,
                                   trainY = Musk_Y,
                                   testX = Musk_x,
                                   testY = Musk_y, 
                                   population =pop ,
                                   obj_function= obj_function,
                                   s_function = sigmoid_function,
                                   binary_function = bin_function)
    
        MuskX_OP, Muskx_OP = dsp()
        print('Number of Features ', MuskX_OP.shape[1])
        
        musk_return = cl.data_classification(train_x = MuskX_OP, train_y = Musk_Y, 
                                      test_x = Muskx_OP ,test_y = Musk_y)
        
        #pd.DataFrame(musk_return).to_json(str(a)+str('_Musk_t4_')+str(obj_function)+'_'+str(sigmoid_function)+'_'+str(bin_function)+str('.csv'), index=False)
        
    for a in range(2):
        print('********************************** AFTER ***********************************')
        print(' --------------------- MADELON DATASET RESULT # --------------------------- ', a)
        print('****************************************************************************')
            
        dsp = pj1.feature_optimization(trainX = Madelon_X,
                                   trainY = Madelon_Y,
                                   testX = Madelon_x,
                                   testY = Madelon_y, 
                                   population =pop ,
                                   obj_function= obj_function,
                                   s_function = sigmoid_function,
                                   binary_function = bin_function)
    
        MadelonX_OP, Madelonx_OP = dsp()
        print('Number of Features ', MadelonX_OP.shape[1])
        
        madelon_return = cl.data_classification(train_x = MadelonX_OP, train_y = Madelon_Y, 
                                      test_x = Madelonx_OP ,test_y = Madelon_y) 
        