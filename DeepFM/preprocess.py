#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Preprocess
import config

import pandas as pd
from itertools import repeat
from sklearn.preprocessing import MinMaxScaler

def get_modified_data(X, all_fields, continuous_fields, categorical_fields, is_bin=False):
    field_dict = dict()
    field_index = []
    X_modified = pd.DataFrame()
    
    for index, col in enumerate(X.columns):
        if col not in all_fields:
            print("{} not included: Check your column list".format(col))
            raise ValueError
            
        if col in continuous_fields:
            scaler = MinMaxScaler()
            
            # whether continuous variables be discretized or not
            if is_bin:
                X_bin = pd.cut(scaler.fit_transform(X[[col]]).reshape(-1, ), config.NUM_BIN, labels=False)
                X_bin = pd.Series(X_bin).astype('str')
                
                X_bin_col = pd.get_dummies(X_bin, prefix=col, prefix_sep='-')
                # field_dict = 1: '0~5', '5~10', ...,
                field_dict[index] = list(X_bin_col.columns)
                # filed_index = 1, 1, 2, 2, ..., m, m, m
                field_index.extend(repeat(index, X_bin_col.shape[1]))
                # 수정된 x; X_modified 에 concat (horizontal direction)
                X_modified = pd.concat([X_modified, X_bin_col], axis=1)
                
            else:
                X_cont_col = pd.DataFrame(scaler.fit_transform(X[[col]]), columns= [col])
                field_dict[index] = col
                field_index.append(index)
                X_modified = pd.concat([X_modified, X_cont_col], axis=1)
                
        if col in categorical_fields:
            X_cat_col = pd.get_dummies(X[col], prefix=col, prefix_sep='-')
            field_dict[index] = list(X_cat_col.columns)
            field_index.extend(repeat(index, X_cat_col.shape[1]))
            X_modified = pd.concat([X_modified, X_cat_col], axis=1)
            
    print('Data Prepared...')
    print('X shape: {}'.format(X_modified.shape))
    print('# of Feature: {}'.format(len(field_index)))
    print('# of Field: {}'.format(len(field_dict)))
    
    return field_dict, field_index, X_modified

