# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 12:33:43 2019

@author: hyunj
"""
import pandas as pd
import numpy as np 
#import matplotlib.pyplot as plt
import glob
import time

import multiprocessing
from functools import partial
from operator import itemgetter

#%% 데이터 불러오기

txtfiles = glob.glob('*.txt')

filename = txtfiles[0]

dataset = pd.read_csv(filename, sep='\t')

#%% 전처리 작업  

dataset = dataset.melt(id_vars = ['time'],
                             var_name = 'div',
                             value_name = 'value')
    
# null 값 제거
dataset = dataset.dropna()

#%%

dataset_div_unique = dataset['div'].str.split('.').str[1:]

dataset_div_unique_list = dataset_div_unique.transform(tuple).unique()


elements = dataset_div_unique_list

n = 1 # N. . .

%time [x[n] for x in elements]


%time zip(*elements)[0]



%time test=  map(itemgetter(1), elements)

dataset_div_unique_list[1]
['.'.join(col).strip() for col in dataset_div_unique_list]

#%%
def test1():
    
    txtfiles = glob.glob('*.txt')

    filename = txtfiles[0]
    
    dataset = pd.read_csv(filename, sep='\t')
    
    dataset = dataset.melt(id_vars = 'time',
                                 var_name = 'div',
                                 value_name = 'value')
        
    # null 값 제거
    dataset = dataset.dropna()
    
    # DIV 분리
    dataset[['ID', 'type', 'index']] = dataset['div'].str.split('.', expand = True)
    
    # DIV 삭제
    dataset.drop(['div'], axis = 'columns', inplace = True)
    
#    dataset.loc[dataset['index'].isnull(),'index'] = '000'
#    
    
    # pivot 
    dataset = dataset.pivot_table(index=['time', 'ID'], columns=['type', 'index'], values=['value'])
    dataset.reset_index(inplace = True)
    dataset.columns = [' '.join(col).strip() for col in dataset.columns.values]

    # 불필요한 열 삭제
    
    dataset.drop(columns = dataset.columns[dataset.columns.str.contains('003|004|005|006', regex=True)], inplace = True)
    
    # 열 이름 수정
    
    return dataset
#%%
def test2():
    
    txtfiles = glob.glob('*.txt')

    filename = txtfiles[0]
    
    dataset = pd.read_csv(filename, sep='\t')
    
    dataset = dataset.melt(id_vars = 'time',
                                 var_name = 'div',
                                 value_name = 'value')
        
    # null 값 제거
    dataset = dataset.dropna()
   
    DATA_LIST = '003|004|005|006'
    
    test = dataset['div'].str.contains(DATA_LIST, regex=True)
    
    dataset = dataset.loc[~test]  
    
    # DIV 분리
    dataset[['ID', 'type', 'index']] = dataset['div'].str.split('.', expand = True)
    
    # DIV 삭제
    dataset.drop(['div'], axis = 'columns', inplace = True)
    
    # pivot 
    dataset = dataset.pivot_table(index=['time', 'ID'], columns=['type', 'index'], values=['value'])
    dataset.reset_index(inplace = True)
    dataset.columns = [' '.join(col).strip() for col in dataset.columns.values]



    return dataset
    
#%%
    
%time test1 = test1()
%time test2 = test2()
