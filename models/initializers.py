#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:19:31 2020

@author: ince
"""


import tensorflow as tf

class LaplacianInizializer(tf.keras.initializers.Initializer):
    
    def __init__(self, L):
        self.L = L
        
    def __call__(self, shape, dtype=None):
        return self.L + tf.random.normal(self.L.shape, mean=0, stddev=10, dtype=dtype)
    
    def get_config(self):  # To support serialization
        return {'L': self.L} 