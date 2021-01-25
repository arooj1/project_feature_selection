# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 00:59:58 2020

@author: quresa9
"""

import numpy as np
import random

class sigmoid_function:
    def __init__(self,function):
        
        functions = {'logistic_function':self.logistic_function,
                     'hyperbolic_function': self.hyperbolic_function,
                     'arctangent_function':self.arctangent_function,
                     'gen_logistic_function':self.gen_logistic_function,
                     'algebraic_function':self.algebraic_function,
                     'gudermannian_function': self.gudermannian_function,
                     'default': self.default_function}
        
        functions.keys()
        #self.x = x
        #self.function_to_load = self.logistic_function
        for f in functions.keys():
            
            if(function ==f):
                self.function_to_load = functions[f]
                print(f)
                break;
            else:
                print('')
             
    def __call__(self,x):
        #print(self.function_to_load())
        return (self.function_to_load(x))
    
    def logistic_function(self,x):
        return 1/ (1+np.exp(x))
            
    def hyperbolic_function(self,x):
        '''
        In term of exponent and reaches infinity 
        '''
        return np.tanh(x)
        
    
    def arctangent_function(self,x):
        return np.arctan(x)
    
    def gen_logistic_function(self,x):
        return (np.power(1+np.exp(x),-2))
        
    def algebraic_function(self,x):
        f_x = (x/np.sqrt(1+np.power(x,2)))
        return f_x
    
    def gudermannian_function(self,x):
        return 2 * np.arctan(np.tanh(x/2))
    
    def default_function(self,x):
        return 1/(1 + np.exp(-2 * x) - 0.25)
    
class jaya_binary:
    def __init__(self,method):
        if method == 'random_r':
            self.method = self.random_r
        elif method == 'default_r':
            self.method = self.default_r

    def __call__(self, p):
        #print(p)
        return self.method(p)
    
    
    def random_r(self,p):
        r = random.uniform(0, 1.0)
        if p > r:
            return 1.0
        else:
            return 0.0
        
    def default_r(self,p):
        if p > 0.5:
            return 1.0
        else:
            return 0.0
            
    
'''
x = 1

a = sigmoid_function(x, 'hyperbolic_function')   
b = sigmoid_function(x, 'logistic_function')  
c = sigmoid_function(x, 'arctangent_function')
d = sigmoid_function(x, 'algebraic_function')   
e = sigmoid_function(x, 'gen_logistic_function')   
f = sigmoid_function(x, 'gudermannian_function')   
 
print('================ TEST CASES =====================')
print('hyperbolic')    
x=a()
print('\n logistic')
y=b()
print('\n arctangent')
z=c()
print('\n algebraic')
p=d()
print('\n gen_logistic')
q=e() 
print('\n gudermannian')
r=f() 
'''