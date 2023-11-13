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

path = 'subject2.csv' #subject.csv에 맞게 바꾸기 !!!
df = pd.read_csv(path)

#Unnamed: 0열 제거하기(csv파일을 만들 때 생긴 열)
df = df.drop(df.columns[0], axis=1)
#subject2은 30개의 열과 229개의 행으로 이루어져 있다.
df.columns.nunique() #30
total_rows = df.shape[0]
total_rows #229
#################################
#3. 데이터 전처리 및 분석 
##3-7. 결측치, 이상치 처리
## 결측치가 있는 열 시각화
columns_with_missing_values = df.columns[df.isnull().any()].tolist()
missing_values = df[columns_with_missing_values].isnull().sum()
missing_values.plot.bar(y=total_rows, color='pink', figsize=(10,10))
plt.title('missing value')
plt.xlabel('columns')
plt.ylabel('total_rows')
plt.show()
## df으로 결측치 직접 확인
missing_values_subset = df[df.isnull().any(axis=1)][columns_with_missing_values]
missing_values_subset
## 결측치가 있는 행의 결측치 수 출력
print(missing_values_subset.isnull().sum(axis=1)) # 결측치가 있는 열이 21개인데 21개가 모두 결측치인 행은 0개



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
##5-4. 시뮬레이션 결과 시각화 & 웹으로 UI 구현해보기


