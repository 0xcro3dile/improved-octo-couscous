#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import helper.py 

"""
Created on sun Feb 6 17:03:19 2022
@author: AhmedAbbas

"""


import numpy as np

"""
GOAL : Find best weights ans baises that minmize the loss fucntion

"""

class NerualNetworks: 
    
    def __init__(self, x, y):
        
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)
            

    def FeedFoward(self):

        self.layer1 = sigmoid(np.dot(self.input  , self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1 , self.weights2))
    
    
    def BackProp(self):

                
        d_weights2  = np.dot(self.layer2.T ,(2*(self.y - self.output) * sigmoid_derv(self.output))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derv(self.output), self.weights2.T) * sigmoid_derv(self.layer1)))

        # weights updating
                                     
        self.weights1 += d_weights1
        self.weights2 += d_weights2




