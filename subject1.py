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
df["location"].describe() # 실내환경인지 확인

#Unnamed: 0열 제거하기(csv파일을 만들 때 생긴 열)
df = df.drop(df.columns[0], axis=1)

#################################
#3. 데이터 전처리 및 분석
##3-5. Null값, 이상치 처리
##3-6. 데이터 전처리 후 분석 정리

#################################
#4. model 만들기
##4-1. 데이터 분리
##4-2. 모델링(선형회귀, 랜덤포레스트, XGBoost,.. )
##4-3. 모델 평가(정확도, RMSE,..)
##4-4. 모델 선정

##. 직접 수집한 데이터 모델 적용해보기

