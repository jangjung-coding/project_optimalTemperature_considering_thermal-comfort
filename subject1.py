'''
Units
Temperature: ˚C
Wind speed: m/s
Heart rate: bpm
Acceleration: m/s2
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = 'subject1.csv' #subject.csv에 맞게 바꾸기 !!!

df = pd.read_csv(path)

#Unnamed: 0열 제거하기(csv파일을 만들 때 생긴 열)
df = df.drop(df.columns[0], axis=1)
df.columns.nunique() #30
#################################
#3. 데이터 전처리 및 분석 
##3-7. 이상치, 결측치 처리
##3-8. 피쳐 엔지니어링(단위와 타입 조정)
##3-9. 피쳐 스케일링
##3-10. 데이터 전처리 및 분석 마무리

#################################
#4. model 만들기
##4-1. 데이터 분리(train, test='therm_sens')
##4-2. 모델 벤치마킹(선형회귀, 랜덤포레스트, XGBoost,.. )
##4-3. 모델 평가(정확도, RMSE,..)
##4-4. 모델 선정

##. 직접 수집한 데이터 모델 적용해보기
#################################
#5. 모델 적용
##5-1. 공간 설계(공간, 구성원 임의 설정)
##5-2. 공간 함수 설계(거리, 옷, thermal sensation을 고려한 함수)
##5-3. 공간 함수로 나온 값이 중앙값, 0이 될때까지 온도 자동 조절(음수면 up, 양수면 down) 시뮬레이션
##5-4. 시뮬레이션 결과 시각화 & 스마트워치 UI 구현해보기


