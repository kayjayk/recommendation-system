#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from layers import FM_layer

tf.keras.backend.set_floatx('float32')

class DeepFM(tf.keras.Model):
    def __init__(self, num_feature, num_field, embedding_size, field_index):
        super(DeepFM, self).__init__()
        self.embedding_size = embedding_size # k: dimensionality of embedding vector
        self.num_feature = num_feature       # f: the num. of all features
        self.num_field = num_field           # m: the num. of grouped field
        self.field_index = field_index       # index which is defined in 'preprocess'
        
        self.fm_layer = FM_layer(num_feature, num_field, embedding_size, field_index)
        
        self.layer1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
        self.layer2 = tf.keras.layers.Dense(units=16, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2)
        self.layer3 = tf.keras.layers.Dense(units=2, activation='relu')
        
        self.final = tf.keras.layers.Dense(units=1, activation='sigmoid')
        
    def __repr__(self):
        return "DeepFM Model: #Field: {}, #Feature: {}, ES: {}".format(
            self.num_field, self.num_feature, self.embedding_size)
    
    def call(self, inputs):
        # 1) FM Component: (num_batch, 2)
        y_fm, new_inputs = self.fm_layer(inputs)
        
        # retrieve Dense Vectors: (num_batch, num_feature*embedding_size)
        new_inputs = tf.reshape(new_inputs, [-1, self.num_feature*self.embedding_size])
        
        # 2) Deep Component
        y_deep = self.layer1(new_inputs)
        y_deep = self.dropout1(y_deep)
        y_deep = self.layer2(y_deep)
        y_deep = self.dropout2(y_deep)
        y_deep = self.layer3(y_deep)
        
        # Concatenation
        y_pred = tf.concat([y_fm, y_deep], 1) # CNN의 Flatten 같은 느낌쓰~
        y_pred = self.final(y_pred) # activated by sigmoid
        y_pred = tf.reshape(y_pred, [-1, ])
        
        return y_pred

