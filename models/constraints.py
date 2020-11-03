#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:55:04 2020

@author: ince
"""
import tensorflow as tf


class CenterAround(tf.keras.constraints.Constraint):
  """Constrains weight tensors to be centered around `ref_value`."""

  def __init__(self, ref_value):
    self.ref_value = ref_value

  def __call__(self, w):
    mean = tf.reduce_mean(w)
    return w - mean + self.ref_value

  def get_config(self):
    return {'ref_value': self.ref_value}