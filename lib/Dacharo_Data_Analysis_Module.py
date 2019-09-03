# -*- coding: utf-8 -*-
"""
2019-08-26 1 Created Ver 0.1
2019-08-26 2 논리 위치 정의 수식 구현
---
2019-09-03 09:51 새로 파일 복구후 정리
           1 병렬처리 수정
@author: hyunj
"""

#%%


import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
from functools import reduce   

#%% 데이터 불러오기


def ReadData(filename):
    
    print("Read Data Start...")     
    
    dataset = pd.read_csv(filename, sep='\t')
    
    print("Read Data End...")     
    
    return dataset


#%% 파일 전처리 


def DataInitialSetting(dataset):
    
    print("Data Initial Setting Start...")     
    
    dataset = dataset.melt(id_vars = ['time'],
                           var_name = 'div',
                           value_name = 'value')
    
    # null 값 제거
    dataset = dataset.dropna()
    
    print("Data Initial Setting End...")    
    
    return dataset

#%% 뽑을 변수 선택(ex. 속도, 가속도 등)
    

def DataSelection(dataset, DATA_LIST_EXIST):
    
    if DATA_LIST_EXIST == True:
        
        # 2018.08.28 DATA LIST
        
        DATA_LIST = ['VehicleUpdate-pos.001',
                     'VehicleUpdate-pos.002',
                     'VehicleUpdate-pos.003',
                     'VehicleUpdate-pos.004',
                     'VehicleUpdate-pos.005',
                     'VehicleUpdate-pos.006',
                     'VehicleUpdate-speed.001',
                     'VehicleUpdate-speed.002',
                     'VehicleUpdate-speed.003',
                     'VehicleUpdate-speed.004',
                     'VehicleUpdate-speed.005',
                     'VehicleUpdate-speed.006',
                     'VehicleUpdate-accel.001',
                     'VehicleUpdate-accel.002',
                     'VehicleUpdate-accel.003',
                     'VehicleUpdate-accel.004',
                     'VehicleUpdate-accel.005',
                     'VehicleUpdate-accel.006',
                     'VehicleUpdate-accelerator',
                     'VehicleUpdate-steeringWheelAngle',
                     'VehicleUpdate-steeringWheelSpeed',
                     'VehicleUpdate-accelerator.1',
                     'VehicleUpdate-brake'
                     ]
        
    else:
            
        dataset_div_unique = dataset['div'].str.split('.').str[1:]
    
        dataset_div_unique_list = dataset_div_unique.transform(tuple).unique()
    
        DATA_LIST = ['.'.join(col).strip() for col in dataset_div_unique_list]
    
    # 데이터 변수 선택(번호)
    
    DATA_LIST_NUM = [0, 1, 6, 7, 12, 13, 19, 20]
    
    DATA_LIST_SELECTED = [DATA_LIST[i] for i in DATA_LIST_NUM]
    
    """
    # DATA_LIST 번호확인
    
    for a, b in enumerate(DATA_LIST, 0):
        print("{}. {}".format(a, b))
    
    # 2019.08.25 DATA LIST 번호
    
    0. VehicleUpdate-pos.001
    1. VehicleUpdate-pos.002
    2. VehicleUpdate-pos.003
    3. VehicleUpdate-pos.004
    4. VehicleUpdate-pos.005
    5. VehicleUpdate-pos.006
    6. VehicleUpdate-speed.001
    7. VehicleUpdate-speed.002
    8. VehicleUpdate-speed.003
    9. VehicleUpdate-speed.004
    10. VehicleUpdate-speed.005
    11. VehicleUpdate-speed.006
    12. VehicleUpdate-accel.001
    13. VehicleUpdate-accel.002
    14. VehicleUpdate-accel.003
    15. VehicleUpdate-accel.004
    16. VehicleUpdate-accel.005
    17. VehicleUpdate-accel.006
    18. VehicleUpdate-accelerator
    19. VehicleUpdate-steeringWheelAngle
    20. VehicleUpdate-steeringWheelSpeed
    21. VehicleUpdate-accelerator.1
    22. VehicleUpdate-brake
    """
    
    DATA_LIST_SELECTED_ = '|'.join(DATA_LIST_SELECTED)
    
    dataset_bool = dataset['div'].str.contains(DATA_LIST_SELECTED_, regex=True)
    
    dataset = dataset.loc[dataset_bool]  
    
    return dataset


#%% 데이터 필터링 필요한 데이터 정리 
    

def DatasetFiltering(dataset):
    
    # DIV 분리 & null 값 처리
    dataset[['ID', 'type', 'index']] = dataset['div'].str.split('.', expand = True)
    dataset.loc[dataset['index'].isnull(),'index'] = '000'
    
    # DIV 삭제
    dataset.drop(['div'], axis = 'columns', inplace = True)
    
    # pivot 
    dataset = dataset.pivot_table(index=['time', 'ID'], columns=['type', 'index'], values=['value'])
    dataset.reset_index(inplace = True)
    dataset.columns = [' '.join(col).strip() for col in dataset.columns.values]

    # ID : str to int
    dataset.ID = dataset.ID.str.slice(1,-1).astype(int)
    
    # 열 이름 수정
    dataset.columns = ['time', 'ID', 'accelX', 'accelY',
                       'posX', 'posY', 'speedX', 'speedY',
                       'steeringangle', 'steeringspeed'
                       ]
    
    # 속도, 가속도 계산
#    dataset.eval('Aaccel =(accelX**2 + accelY**2)**(1/2)', inplace = True)
#    dataset.eval('Aspeed =(speedX**2 + speedY**2)**(1/2)', inplace = True)

    dataset.eval('Aaccel = accelX', inplace = True)
    
    dataset.eval('Aspeed = speedX * 3.6', inplace = True)    
    dataset.drop(dataset.iloc[:,[2,3,6,7]], axis = 1, inplace = True)

    return dataset

#%% 절대 공간 범위 데이터 추출


def ExtractAbsCoorRangeData(dataset, XYrange):
    
    Xmin, Xmax, = XYrange['Xmin'], XYrange['Xmax']
    Ymin, Ymax, = XYrange['Ymin'], XYrange['Ymax'] 
    
    XCond = (Xmin <= dataset.posX) & (dataset.posX <= Xmax)
    YCond = (Ymin <= dataset.posY) & (dataset.posY <= Ymax)
    
    return dataset[XCond & YCond]
    
#%% 논리 위치 정의
    

def DefineLogicalCoordinate(dataset, XYrange) :
    
    # 논리 위치 범위 
    Xmin, Xcen, Xmax, Xint = XYrange['Xmin'], XYrange['Xcen'], XYrange['Xmax'], XYrange['Xint']
    Ymin, Ycen, Ymax, Yint = XYrange['Ymin'], XYrange['Ycen'], XYrange['Ymax'], XYrange['Yint']
    
    Yimin = (Ycen - Ymin) // Yint
    Yimax = (Ycen - Ymax) // Yint
    
    Ximin = (Xcen - Xmin) // Xint
    Ximax = (Xcen - Xmax) // Xint
    
    Ybins = range(Ycen - Yimin * Yint, Ycen - Yimax * Yint, Yint)
    Xbins = range(Xcen - Ximin * Xint, Xcen - Ximax * Xint, Xint)
    
    Ylabels = list(Ybins)
    Xlabels = list(Xbins)
    
    dataset['Ybin'] = pd.cut(dataset.posY, bins = Ylabels, labels = Ylabels[:-1])
    dataset['Xbin'] = pd.cut(dataset.posX, bins = Xbins, labels = Xlabels[:-1])
    
    return dataset


#%% 차량찾기
    

def FindMyVehicleInfo(dataset):
    
    print("Find My Vehicle Infomation Start...")  
    
    my_vehicle_dataset = dataset[dataset.ID == 0]
    
    print("Find My Vehicle Infomation End...")  
    
    return my_vehicle_dataset

#%%
    

def FindOtherVehicleInfo(dataset):
    
    print("Find Other Vehicle Infomation Start...")  
    
    other_vehicle_dataset = dataset[dataset.ID != 0]
    
    print("Find Other Vehicle Infomation End...")  
    
    return other_vehicle_dataset


#%%
def FindNearVehicleRank(my_vehicle_index, my_vehicle_time, Xbin0, Ybin0, Xpos0, Ypos0, Aspeed, dataset, side_lane, XYrange):

    my_index, my_time, Xbin, Ybin, Xpos, Ypos, Aspeed = my_vehicle_index, my_vehicle_time, Xbin0, Ybin0, Xpos0, Ypos0, Aspeed
    
#    for debugging
#    my_time, Xpos, Ypos, Aspeed = my_vehicle_dataset.time.iloc[100], my_vehicle_dataset.posX.iloc[100],  my_vehicle_dataset.posY.iloc[100], my_vehicle_dataset.Aspeed.iloc[100]
    
    sametimedata = dataset[(dataset.time == my_time) & (Xpos - side_lane  <= dataset.posX) & (dataset.posX <= Xpos + side_lane)]
    
    sametimedata.eval('distance = ((posX - @Xpos)**2 + (posY - @Ypos)**2)**(1/2)', inplace = True)
    
    sametimedata.loc[:, 'rank'] = sametimedata['distance'].rank(ascending = True)
    
#    dataset.loc[:, dataset.posY >= Ypos].eval('TTC = distance/@Aspeed', inplace = True)
    
    sametimedata.loc[:,'TTC'] = np.nan
    
    TTC1 = sametimedata.loc[(Ypos              >= sametimedata.posY) &
                            (Xpos - side_lane  <= sametimedata.posX) &
                            (sametimedata.posX <= Xpos + side_lane)
                            ]
    
    TTC2 = sametimedata.loc[(Ypos              <  sametimedata.posY) &
                            (Xpos - side_lane  <= sametimedata.posX) &
                            (sametimedata.posX <= Xpos + side_lane)
                            ]
    
    # 하행 +, 상행 - (상행 후처리 필요)
    TTC1.eval('TTC = (distance/((@Aspeed-Aspeed)/3.6))', inplace = True)
    
    TTC2.eval('TTC = (distance/((Aspeed-@Aspeed)/3.6))', inplace = True)
    
    sametimedata = pd.concat([TTC1, TTC2])
    
    return sametimedata    
#%%
def ParalleRunsRank(data_list, dataset, side_lane, XYrange):
    
    print("Paralle Runs Start...")  
    
    poolrank = multiprocessing.Pool(processes = 4)
    
    pool_test = partial(FindNearVehicleRank, dataset = dataset , side_lane = side_lane, XYrange = XYrange)
    
    result = poolrank.starmap(pool_test, data_list) 
    
    print("Paralle Runs End...")  
    
    return result


#%% result list to dataframe

def ResultToDF(result):
    
    print("Data Join Start...")       
    
    result = list(i for i in result if len(i) == 18)
    
    result = np.array(result)
    
    nearVehiclesID = pd.DataFrame(result[:,1:], index = result[:,0], columns=['time', 'front','rear', 'left_front', 'left_rear', 'right_front', 'right_rear', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'])

    print("Data Join End...")       
    
    return nearVehiclesID

def ResultToDFRank(resultRank):
    
    print("Data Join Start...")       
    
    resultRank = pd.concat(resultRank)

    print("Data Join End...")       
    
    return resultRank
#%% 주변차량 데이터셋 변환 및 Join
 

def DataJoin(dataset, my_vehicle_dataset, nearVehiclesID):
    
    JoinedDataset_0 = pd.merge(my_vehicle_dataset, nearVehiclesID, left_index = True, right_index = True, on = 'time')
    
    JoinedDataset_0.columns[10:]
    
    test = list(map(lambda i : pd.merge(nearVehiclesID[['time', i]], dataset.add_suffix('_' + i), left_index = True, how = 'left',
                                        left_on = ['time', i], right_on = ['time_' + i, 'ID_' + i]).drop(columns = ['time_' + i, i]),
                                        JoinedDataset_0.columns[10:]))
    test0 = reduce(lambda left, right: pd.merge(left, right, on = 'time'), test)    
        
    test1 = pd.merge(my_vehicle_dataset, test0, on = 'time')
    
    return test1


    

#%% 데이터 저장

def DataToCSV(dataset, filename):
    
    print("Data to CSV Start...")  
    
    csv_filename = filename.split('.')[0] + '.csv'
    dataset.to_csv(csv_filename, index = False)
    
    print(csv_filename)
    
    print("Data to CSV End...")  