# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 00:59:58 2020

@author: quresa9
"""

import numpy as np

class sigmoid_function:
    def __init__(self, x, function):
        functions = {'logistic_function':self.logistic_function,
                     'hyperbolic_function': self.hyperbolic_function,
                     'arctangent_function':self.arctangent_function,
                     'gen_logistic_function':self.gen_logistic_function,
                     'algebraic_function':self.algebraic_function}
        
        functions.keys()
        self.x = x
        #self.function_to_load = self.logistic_function
        for f in functions.keys():
            print(f)
            if(function ==f):
                self.function_to_load = functions[f]
                break;
            else:
                print('function not available')
             
    def __call__(self):
        print(self.function_to_load())
        return (self.function_to_load())
    
    def logistic_function(self):
        return 1/ (1+np.exp(-self.x))
            
    def hyperbolic_function(self):
        return np.tanh(self.x)
        
    
    def arctangent_function(self):
        return np.arctan(self.x)
    
    def gen_logistic_function(self):
        return (np.power(1+np.exp(self.x),3))
        
    def algebraic_function(self):
        f_x = (self.x/np.sqrt(1+np.power(x,2)))
        return f_x
    
    

x = 33
a = sigmoid_function(x, 'hyperbolic_function')   
b = sigmoid_function(x, 'logistic_function')  
c = sigmoid_function(x, 'arctangent_function')
d = sigmoid_function(x, 'algebraic_function')   
e = sigmoid_function(x, 'gen_logistic_function')   
 
    
x=a()
y=b()
z=c()
p=d()
q=e()     