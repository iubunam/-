# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 23:35:16 2019

@author: hyunj
"""

import os
import glob
import pandas as pd
from scipy import stats
import numpy as 
#%%
files = glob.glob('DataFiles/*.csv')

dataset_total = pd.concat([pd.read_csv(fp).assign(New=os.path.basename(fp).split('.')[0]) for fp in files])

dataset_total[['P_ID', 'S_Direction', 'S_Peak', 'S_Way']] = dataset_total['New'].str.split(" ", expand = True)

dataset_total.drop(columns = 'New', inplace = True)


#%% 

# 방향별 TTC정리
dataset_total.loc[dataset_total['S_Direction'] == 'Sangheng', 'TTC'] = - dataset_total.loc[dataset_total['S_Direction'] == 'Sangheng', 'TTC']

# TTC 마이너스값 처리
dataset_total.loc[dataset_total['TTC'] < 0, 'TTC'] = 999

#%% 이상치 제거
# 이상치 찾기
def FindOutlier(dataset_total, var, sigma):
    outlierdataset = dataset_total.loc[(np.abs(stats.zscore(dataset_total[var])) > sigma), var]
    return outlierdataset

# 이상치 -> nan
def FindOutlierNan(dataset_total, var, sigma):
    dataset_total.loc[(np.abs(stats.zscore(dataset_total[var])) > sigma), var] = np.nan

for i in dataset_total.columns[4:8]:
    FindOutlierNan(dataset_total, i, 3)


#%%
# 데이터 Export
dataset_total.to_csv('dataset_total.csv', index = False)
 





