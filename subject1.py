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
from sklearn.impute import KNNImputer
from scipy import stats

path = 'subject1.csv' #subject.csv에 맞게 바꾸기 !!!
df = pd.read_csv(path)

#Unnamed: 0열 제거하기(csv파일을 만들 때 생긴 열)
df = df.drop(df.columns[0], axis=1)

#subject1은 24개의 열과 146개의 행으로 이루어져 있다.
df.columns.nunique() #24
total_rows = df.shape[0]
total_rows #146
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
df[df.isnull().any(axis=1)][columns_with_missing_values]
'''
결측치 열 21개
    'mean.Temperature_60',
    'grad.Temperature_60', 
    'sd.Temperature_60',
    'mean.Humidity_60', 
    'grad.Humidity_60',
    'sd.Humidity_60',
    'mean.Winvel_60',
    'grad.Winvel_60',
    'sd.Winvel_60',
->'Temperature', 'Humidity', 'Winvel'은 2주 사이에 큰 차이가 없어 평균값으로 결측치 처리
    'mean.Solar_60',
    'grad.Solar_60',
    'sd.Solar_60',
->'Solar'는 시간에 영향을 많이 받기에 'Vote_time'에 맞춰서 결측치 처리
    'mean.hr_60',
    'grad.hr_60',
    'sd.hr_60',
    'mean.WristT_60',
    'grad.WristT_60',
    'sd.WristT_60',
    'mean.PantT_60',
    'grad.PantT_60',
    'sd.PantT_60'
-> 생리학적 변수 'hr', 'WristT', 'PantT'는 비교적 순간적으로 급격한 변화를 띄지 않는 변수로 주변 10개 평균값으로 결측치 처리
'''
## -> 'Temperature', 'Humidity', 'Winvel'은 2주 사이에 큰 차이가 없어 평균값으로 결측치 처리
environmental_features = ['mean.Temperature_60', 'grad.Temperature_60', 'sd.Temperature_60', 'mean.Humidity_60', 'grad.Humidity_60', 'sd.Humidity_60', 'mean.Winvel_60', 'grad.Winvel_60', 'sd.Winvel_60']
df[environmental_features] = df[environmental_features].fillna(df[environmental_features].mean())

## -> 'Solar'는 시간에 영향을 매우 많이 받기에 'Vote_time'에 맞춰서 결측치 처리
## string 타입의 'Vote_time' 열을 분 단위로 계산하여 int64 타입으로 'Vote_time_as_number' 열에 추가. 2주라 짧은 시간이므로 일 단위는 무시
## 예를 들어, 하루를 1440이라고 보면 오전 9시는 540, 15시(오후3시)는 900이 된다
df['Vote_time_as_number'] = pd.to_datetime(df['Vote_time'], format='%m/%d/%Y %H:%M').dt.hour * 60 + pd.to_datetime(df['Vote_time'], format='%m/%d/%Y %H:%M').dt.minute

plt.scatter(x=df[ 'Vote_time_as_number'], y=df['mean.Solar_60'], alpha=0.7) #그림을 보면 600~1000 즉, 10시~17시 해가 떠있을때 Solar가 큰것을 볼 수 있음 
plt.xlabel('Vote_time_as_number')
plt.ylabel('mean.Solar_60')
plt.show()
## 따라서 KNNImputer를 이용해 'Vote_time_as_number'이 비슷한 주변을 찾아 결측치 처리가 가능함
imputer = KNNImputer(n_neighbors=3)
df[['Vote_time_as_number', 'mean.Solar_60']] = pd.DataFrame(imputer.fit_transform(df[['Vote_time_as_number', 'mean.Solar_60']]), columns=['Vote_time_as_number', 'mean.Solar_60'])
df[['Vote_time_as_number', 'grad.Solar_60']] = pd.DataFrame(imputer.fit_transform(df[['Vote_time_as_number', 'grad.Solar_60']]), columns=['Vote_time_as_number', 'grad.Solar_60'])
df[['Vote_time_as_number', 'sd.Solar_60']] = pd.DataFrame(imputer.fit_transform(df[['Vote_time_as_number', 'sd.Solar_60']]), columns=['Vote_time_as_number', 'sd.Solar_60'])

## -> 생리학적 변수 'hr', 'WristT', 'PantT'는 비교적 순간적으로 급격한 변화를 띄지 않는 변수로 결측치 앞뒤로 5개씩, 총 10개의 평균값으로 결측치 처리
physiological_features = ['mean.hr_60', 'grad.hr_60', 'sd.hr_60', 'mean.WristT_60', 'grad.WristT_60', 'sd.WristT_60','mean.PantT_60', 'grad.PantT_60', 'sd.PantT_60']
df[physiological_features] = df[physiological_features].fillna(df[physiological_features].rolling(window=11, min_periods=1, center=True).mean())
## missing_values가 있는지 확인
print(df.columns[df.isnull().any()].tolist())
'''
'mean.WristT_60',
'grad.WristT_60',
'sd.WristT_60',
'mean.PantT_60',
'grad.PantT_60',
'sd.PantT_60'
'''
df[physiological_features] = df[physiological_features].fillna(df[physiological_features].rolling(window=11, min_periods=1, center=True).mean()) #다시 한번 결측치 처리
## 결측치 최종 확인
print(df.columns[df.isnull().any()].tolist()) #결측치 없음!!!

## 이상치 확인
## 공간이동에 따른 순간적인 환경변화, 운동, 식사 등에 따른 생리학적 변수의 변화로 인한 이상치가 발생할 수 있고 이런것들도 모델에 학습시키기 위해 이상치를 제거를 최소화 하기로 결정
df.describe()
df.boxplot(rot=90, figsize=(10,10))
## 환경변수('Temperature', 'Humidity', 'Winvel', 'Solar')에 대한 이상치 확인 -> 외부 데이터(기상청)를 가져온 것이므로 이상치 제거하지 않기로 결정
environmental_features_include_solar = ['mean.Temperature_60', 'grad.Temperature_60', 'sd.Temperature_60', 'mean.Humidity_60', 'grad.Humidity_60', 'sd.Humidity_60', 'mean.Winvel_60', 'grad.Winvel_60', 'sd.Winvel_60', 'mean.Solar_60', 'grad.Solar_60', 'sd.Solar_60']
plt.boxplot(df[environmental_features_include_solar]) # 'Solar'에서 이상치가 많이 발견됨. 이는 subject1 측정 계절이  12월, 즉 겨울 샌프란시스코 일간 온도차가 크기 때문으로 판단
## 생리학적 변수 'hr', 'WristT', 'PantT'에 대한 이상치는 순간적인 변화에 의해 생기는 값이므로 개인 therml_sens를 찾는 회귀, 분류 모델이 영향을 많이 줄 것이므로 평균값으로 대체하기로 결정
physiological_features = ['mean.hr_60', 'grad.hr_60', 'sd.hr_60', 'mean.WristT_60', 'grad.WristT_60', 'sd.WristT_60', 'mean.PantT_60', 'grad.PantT_60', 'sd.PantT_60']
df[physiological_features].boxplot(rot=90, figsize=(10,10))
## Z-score로 이상치 찾고 평균값으로 대체
z = np.abs(stats.zscore(df[physiological_features]))
print(np.where(z > 3)) # 표준정규분포에서 약 99.7%의 데이터가 평균에서 ±3 표준편차 범위에 위치하는 outlier를 이상치로 판단
df[physiological_features] = df[physiological_features][(z < 3).all(axis=1)] #이상치 제거
df[physiological_features] = df[physiological_features].fillna(df[physiological_features].mean()) #이상치 제거 후 결측치 처리
## 이상치 최종 확인
df[physiological_features].boxplot(rot=90, figsize=(10,10)) #이상치 처리 끝!!!

##3-8. 피쳐 스케일링(RobustScaler)
## RobustScaler를 사용하면 모든 변수들이 같은 스케일을 갖게 되며, StandardScaler에 비해 스케일링 결과가 더 넓은 범위로 분포하게 됨
## 따라서 StandardScaler에 비해 이상치의 영향이 적어진다는 장점이 있는데 이는 모델의 과대적합을 방지하고 이상치 제거를 최소화한 앞선 작업과도 연결됨(줄102에서 이상치 제거를 최소화 했어서 RobustScaler를 사용하기로 결정)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df[environmental_features_include_solar] = scaler.fit_transform(df[environmental_features_include_solar]) #환경변수 스케일링
df[physiological_features] = scaler.fit_transform(df[physiological_features]) #생리학적 변수 스케일링
## 스케일링 확인
df[environmental_features_include_solar].describe()
df[environmental_features_include_solar].boxplot(rot=90, figsize=(10,10))
df[physiological_features].describe()
df[physiological_features].boxplot(rot=90, figsize=(10,10))
## -> 차원축소를 위해 minmaxscaler로 데이터 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[environmental_features_include_solar] = scaler.fit_transform(df[environmental_features_include_solar]) #환경변수 0~1 스케일링
df[physiological_features] = scaler.fit_transform(df[physiological_features]) #생리학적 변수 0~1 스케일링
## 스케일링 확인
df[environmental_features_include_solar].describe()
df[environmental_features_include_solar].boxplot(rot=90, figsize=(10,10))
df[physiological_features].describe()
df[physiological_features].boxplot(rot=90, figsize=(10,10))
## Other features(기타 변수) 4개('ID','Vote_time', 'Vote_time_as_number')는 스케일링 하지 않음
## 'therm_sens'는 우리가 구하고자 하는 변수(label_y)이므로 스케일링 하지 않음

##3-9. 차원 축소
## 'therm_sens'를 잘 예측하고 차원의 저주를 피하기 위해 비슷한 성격의 변수들로 새로운 변수를 만들어 차원 축소를 진행하기로 결정
## Environmental features(환경 변수) 12개
pd.plotting.scatter_matrix(df[environmental_features_include_solar], alpha=0.8, figsize=(15, 15), diagonal='kde')
## Temperature, Humidity, Winvel, Solar의 4개 변수에서 mean, grad, sd로 새로운 변수를 만들어 관계 파악
mean_environmental_features = ['mean.Temperature_60', 'mean.Humidity_60', 'mean.Winvel_60', 'mean.Solar_60']
grad_environmental_features = ['grad.Temperature_60', 'grad.Humidity_60', 'grad.Winvel_60', 'grad.Solar_60']
sd_environmental_features = ['sd.Temperature_60', 'sd.Humidity_60', 'sd.Winvel_60', 'sd.Solar_60']
## mean_environmental_features 상관관계 시각화
pd.plotting.scatter_matrix(df[mean_environmental_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
pd.plotting.scatter_matrix(df[grad_environmental_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
pd.plotting.scatter_matrix(df[sd_environmental_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
## -> mean, grad, sd으로 나누니까 선형적인 상관관계가 아닌 군집화된 데이터 특성이 보임
Temperature_features = ['mean.Temperature_60', 'grad.Temperature_60', 'sd.Temperature_60']
Humidity_features = ['mean.Humidity_60', 'grad.Humidity_60', 'sd.Humidity_60']
Winvel_features = ['mean.Winvel_60', 'grad.Winvel_60', 'sd.Winvel_60']
Solar_features = ['mean.Solar_60', 'grad.Solar_60', 'sd.Solar_60']
pd.plotting.scatter_matrix(df[Temperature_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
pd.plotting.scatter_matrix(df[Humidity_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
pd.plotting.scatter_matrix(df[Winvel_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
pd.plotting.scatter_matrix(df[Solar_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
## -> Temperature, Humidity, Winvel, Solar로 나누니까 비선형적, 군집화 등 특별한 규칙이 보이지 않음
## 따라서 환경변수 12개를 각각의 mean, grad, sd으로 나누어 차원 축소를 진행하기로 결정 (12개 -> 3개)
## t-SNE (t-distributed Stochastic Neighbor Embedding)를 활용한 차원축소
from sklearn.manifold import TSNE


## Physiological features(생리학적 변수) 9개
pd.plotting.scatter_matrix(df[physiological_features], alpha=0.8, figsize=(15, 15), diagonal='kde')

## 'Vote_time', 'Vote_time_as_number'에서 'Vote_time_as_number'만 사용하기로 결정

'''
- Environmental features(환경 변수) 12개 -> 3개
        'mean.Temperature_60', 'grad.Temperature_60', 'sd.Temperature_60', 
        'mean.Humidity_60', 'grad.Humidity_60', 'sd.Humidity_60', 
        'mean.Winvel_60', 'grad.Winvel_60', 'sd.Winvel_60', 
        'mean.Solar_60', 'grad.Solar_60', 'sd.Solar_60',
- Physiological features(생리학적 변수) 9개 -> 3개
        'mean.hr_60', 'grad.hr_60', 'sd.hr_60', 
        'mean.WristT_60', 'grad.WristT_60', 'sd.WristT_60', 
        'mean.PantT_60', 'grad.PantT_60', 'sd.PantT_60'
- Comfortability features(쾌적성 변수) 1개
        'therm_sens'
- Other features(기타 변수) 3개 -> 1개
        'ID', 'Vote_time', 'Vote_time_as_number' 
-> 총 X 피쳐 갯수 7개, 총 Y 피쳐 갯수 1개
'''
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






