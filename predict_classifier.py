import xgboost as xgb

import numpy as np
import pandas as pd
import re
import os
from scipy.sparse import csr_matrix

op_set = ['non', 'conv2d1x1', 'conv2d2x2', 'conv2d3x3', 'conv2d5x5', 'conv2d7x7', 'matmul','relu','leakyrelu','sigmoid','softmax','biasadd','batchnorm','pad','slice','split','concat','reshape','upsample','add','mul','exp','sub','bninference','scale','eltwise']
op_set_new = ['non', 'pad', 'conv2d1x1', 'conv2d2x2', 'conv2d3x3', 'conv2d5x5', 'conv2d7x7', 'matmul', 'stridedslice', 'slice','add','mul','exp','sub', 'biasadd','eltwise','batchnorm','bninference','scale',  'relu','leakyrelu','sigmoid','softmax','split','concat','reshape','upsample']

def newindex_transfer(col_list, data_list):
    for i in range(len(col_list)):
        if col_list[i] <= 50:
            data_list[i] = op_set_new.index(op_set[data_list[i]])
    return col_list, data_list

def simplify_data(col_list, data_list):
    # 解析numpy array
    values_dict = dict(zip(col_list, data_list))
    
    # 修改特定序号的值
    zeros_indices = [57, 64, 71, 78, 85, 86, 87, 88, 89, 90, 91, 92, 93]
    for index in zeros_indices:
        if index in values_dict:
            values_dict[index] = 0
    
    # 检查序号 51，并可能调换序号 52～55 的值
    desc_start = [51, 58, 65, 72, 79]
    for index in desc_start:
        if index in values_dict and values_dict[index] == 2: # NHWC->NCHW
            swap_indices = [index+2, index+3, index+4]
            values_to_swap = [values_dict[i] for i in swap_indices]
            values_dict[index+2] = values_to_swap[2]
            values_dict[index+3] = values_to_swap[0]
            values_dict[index+4] = values_to_swap[1]
        elif index in values_dict and values_dict[index] == 4: # HWCN->NCHW
            swap_indices = [index+1, index+2, index+3, index+4]
            values_to_swap = [values_dict[i] for i in swap_indices]
            values_dict[index+1] = values_to_swap[3]
            values_dict[index+2] = values_to_swap[2]
            values_dict[index+3] = values_to_swap[0]
            values_dict[index+4] = values_to_swap[1]
        values_dict[index] = 0
        
    # 删除不需要的序号
    deleted_indices = [x for x in range(11, 51)] + zeros_indices + desc_start
    old_keys = [x for x in range(1, 96) if x not in deleted_indices]
    new_values_dict = {}
    new_key = 1
    for old_key in old_keys:
        if old_key in values_dict:
            new_values_dict[new_key] = values_dict[old_key]
        else:
            new_values_dict[new_key] = 0
        new_key += 1
        
    # 生成新的numpy array
    new_col_list = np.array(list(new_values_dict.keys()))
    new_data_list = np.array(list(new_values_dict.values()))
    return new_col_list, new_data_list
    
def parse_input_string(input_str):
    # 使用正则表达式匹配冒号前后的数字
    matches = re.findall(r'(\d+):(\d+)', input_str)

    # 提取冒号前后的数字并转换为两个NumPy数组
    numbers_before_colon = np.array([int(match[0]) for match in matches])
    numbers_after_colon = np.array([int(match[1]) for match in matches])
    numbers_before_colon, numbers_after_colon = newindex_transfer(numbers_before_colon, numbers_after_colon)
    numbers_before_colon, numbers_after_colon = simplify_data(numbers_before_colon, numbers_after_colon)

    return numbers_before_colon, numbers_after_colon
      
model = xgb.Booster(model_file='classifiermodel.model')
data_x = "1:3 2:23 3:24 4:8 5:1 51:2 52:1 53:14 54:14 55:512 56:1 57:2 58:4 59:3 60:3 61:512 62:512 63:1 64:2 65:3 66:512 67:1 68:1 69:1 70:1 71:1 72:4 73:1 74:1 75:512 76:2048 77:1 78:2 79:3 80:2048 81:1 82:1 83:1 84:1 85:1 86:2 87:6 "
col, data = parse_input_string(data_x)
print(col)
print(data)
row = np.zeros(col.size)
csr = csr_matrix((data, (row, col)), shape=(1, 38)).toarray()

x_test = xgb.DMatrix(csr)
pred_y=model.predict(x_test)

print(pred_y)
'''
# 找到53和54标签在col数组中的索引
index_53 = np.where(col == 53)[0][0]
index_54 = np.where(col == 54)[0][0]

# 在循环中更改53和54的值
value_53 = 1024
while value_53 >= 8:
    data_copy = data.copy()
    data_copy[index_53] = value_53
    data_copy[index_54] = value_53
        
    csr = csr_matrix((data_copy, (np.zeros(col.size), col)), shape=(1, 98)).toarray()
    x_test = xgb.DMatrix(csr)
    pred_y = model.predict(x_test)
        
    print(f"Value 53: {value_53}, Value 54: {value_53}, Prediction: {pred_y}")
    value_53 = value_53/2
'''
