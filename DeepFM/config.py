#!/usr/bin/env python
# coding: utf-8

# In[ ]:


ALL_FIELDS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race',
             'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'country']
CONT_FIELDS = ['age', 'fnlwgt', 'education-num',
               'capital-gain', 'capital-loss', 'hours-per-week']
CAT_FIELDS = list(set(ALL_FIELDS).difference(CONT_FIELDS))

# Hyperparams
NUM_BIN = 10
BATCH_SIZE = 256
EMBEDDING_SIZE = 10

