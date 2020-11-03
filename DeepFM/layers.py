#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

class FM_layer(tf.keras.layers.Layer): # inheritance
    def __init__(self, num_feature, num_field, embedding_size, field_index):
        super(FM_layer, self).__init__()
        self.embedding_size = embedding_size # k: dimensionality of embedding vector
        self.num_feature = num_feature       # f: the num. of all features
        self.num_field = num_field           # m: the num. of grouped field
        self.field_index = field_index       # index which is defined in 'preprocess'
        
        # Parameters of FM layer
        # w: capture 1st order interactions
        # V: capture 2nd order interactions
        self.w = tf.Variable(tf.random.normal(shape=[num_feature],
                                             mean=0.0, stddev=1.0), name='w')
        self.V = tf.Variable(tf.random.normal(shape=(num_field, embedding_size),
                                             mean=0.0, stddev=0.01), name='V')
        
    def call(self, inputs):
        x_batch = tf.reshape(inputs, [-1, self.num_feature, 1])
        # param V를 field_index에 맞게 복사하여, num_feature에 맞게 늘림 (embeds = 늘린 V)
        embeds = tf.nn.embedding_lookup(params=self.V, ids=self.field_index)
        
        # input to be used in Deep Component
        # (batch_size, num_feature, embedding_size)
        # think about 'broadcasting'!! ex) (256, 108, 1) x (108, 10) => (256, 108, 10)
        # new_inputs 이 FM 에서의 xv의 역할
        new_inputs = tf.math.multiply(x_batch, embeds)
        
        # (batch_size, )
        linear_terms = tf.reduce_sum(tf.math.multiply(self.w, inputs), axis=1, keepdims=False)
        
        # (batch_size, )
        interactions = 0.5 * tf.subtract(
                        tf.square(tf.reduce_sum(new_inputs, [1, 2])),
                        tf.reduce_sum(tf.square(new_inputs), [1, 2])
                        )
        
        linear_terms = tf.reshape(linear_terms, [-1, 1])
        interactions = tf.reshape(interactions, [-1, 1])
        
        y_fm = tf.concat([linear_terms, interactions], 1)
        
        return y_fm, new_inputs

