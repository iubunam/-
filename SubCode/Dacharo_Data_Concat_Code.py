# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 23:35:16 2019

@author: hyunj
"""

import os
import glob
import pandas as pd

#%%
files = glob.glob('DataFiles/*.csv')

dataset_total = pd.concat([pd.read_csv(fp).assign(New=os.path.basename(fp).split('.')[0]) for fp in files])

dataset_total[['P_ID', 'S_Direction', 'S_Peak', 'S_Way']] = dataset_total['New'].str.split(" ", expand = True)

dataset_total.drop(columns = 'New', inplace = True)


#%%


#
#dataset_total.loc[dataset_total['S_Way'] == '(2)', 'S_Way'] = 'M-Hipass'
#
#dataset_total.loc[dataset_total['S_Way'] == '(3)', 'S_Way'] = 'TCS'
#
#dataset_total.loc[dataset_total['S_Way'] == '(1)', 'S_Way'] = 'Right


dataset_total.loc[dataset_total['S_Direction'] == 'Sangheng', 'TTC'] = - dataset_total.loc[dataset_total['S_Direction'] == 'Sangheng', 'TTC']

dataset_total.loc[dataset_total['TTC'] < 0, 'TTC'] = 999

dataset_total.to_csv('dataset_total.csv', index = False)
 
