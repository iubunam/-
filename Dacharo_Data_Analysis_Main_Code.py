""" Version Commentary

2019-08-26  데이터 셋 구분 필요
            - Raw Data Set -> Grid 단위로 : 네트워크 분
            - 실험차량 데이터 + 주변차량 ID + 주변차량 데이터 : 주행거동분석

2019-08-26 1.2.1 실험차량 주변차량 정의
                 1) 실험차량 공간범위 정의
                 2) 공간범위내 차량 가까운 순으로 10대 ID 추출
"""

""" 확인사항
- 가속도 계산 방법

"""
#%% 모듈 불러오기


import glob
import time
from lib import 'Dacharo_Data_Analysis_Module_V2.0' as DA


#%% Initial Variable Setting


# Txt 파일 불러오기
txtfiles = glob.glob('Data_Files/*.txt')

# 차로폭 설정
side_lane = 1.8 + 3.6 + 3.6

XYrange = dict(Xmin = -14515, Xcen = -14443, Xmax = -14341, Xint = 5,
              Ymin = 6646, Ycen = 7146, Ymax = 7595, Yint = 100)

# 전체 시간 측정 
total_time = time.time()

DEBUGGING_MODE = True
#DEBUGGING_MODE = False 


#%% Main Code


if __name__ == '__main__':        
    
    # Debuggin Mode
    if DEBUGGING_MODE == True:
            
        txtfiles = list([txtfiles[0], txtfiles[3]])
            
            
    for filename in txtfiles:
        
        print(filename)
        # 1 File 처리 시간 계산
        code_start_time = time.time()
       
        print('----- ',filename,' ----- START -----')

        """ 전체 데이터 분석 """
        
        # 파일 불러오기
        dataset = DA.ReadData(filename)
        
        # 파일 전처리
        dataset = DA.DataInitialSetting(dataset)
        
        # 뽑을 데이터 선택
        dataset = DA.DataSelection(dataset, DATA_LIST_EXIST = True)
        
        # 데이터 필터링 필요한 데이터 정리(Pivot, 라벨링, 속도, 가속도 계산)
        dataset = DA.DatasetFiltering(dataset)
        
        # 절대 공간 범위 데이터 추출
        dataset = DA.ExtractAbsCoorRangeData(dataset, XYrange)
                    
        # 논리 좌표 변환
        dataset = DA.DefineLogicalCoordinate(dataset, XYrange)
        
        # 1차 가공 처리 완료 파일 저장
        DATASET_1 = dataset
                
        """ 실험차량 분석 """
        
        # 실험차량 데이터셋 추출
        my_vehicle_dataset = DA.FindMyVehicleInfo(dataset)
        
        # 다른 차량 데이터 셋 추출
        other_vehicle_dataset = DA.FindOtherVehicleInfo(dataset)
        
        # Zip
        
        data_list = zip(my_vehicle_dataset.index ,
                        my_vehicle_dataset.time  ,
                        my_vehicle_dataset.Xbin ,
                        my_vehicle_dataset.Ybin ,
                        my_vehicle_dataset.posX ,
                        my_vehicle_dataset.posY,
                        my_vehicle_dataset.Aspeed
                        )
        
        # 병렬처리
        resultRank = DA.ParalleRunsRank(data_list, dataset, side_lane, XYrange)
        
        JoinedDatasetRank = DA.ResultToDFRank(resultRank)
        
        DA.DataToCSV(JoinedDatasetRank, filename)   
                
        
        
        print("--- A file takes %s seconds ---" % (time.time() - code_start_time))

    print("--- Total takes %s seconds ---" % (time.time() - total_time))

