#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:55:04 2020

@author: ince
"""
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
import tensorflow as tf

class LaplacianConstraint(tf.keras.constraints.Constraint):

  def __init__(self, alpha):
      self.alpha = alpha

  def __call__(self, w):
    trace = tf.linalg.trace(w)
    return w * (
              math_ops.cast(math_ops.greater_equal(trace, 1.), K.floatx()) 
             * math_ops.cast(math_ops.less_equal(w - tf.linalg.diag_part(w), 0), K.floatx()) 
            )  + self.alpha * tf.norm(w, 1) 
            
            