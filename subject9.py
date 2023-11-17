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

path = 'subject9.csv' #subject.csv에 맞게 바꾸기 !!!
df = pd.read_csv(path)

#Unnamed: 0열 제거하기(csv파일을 만들 때 생긴 열)
df = df.drop(df.columns[0], axis=1)
#subject9은 15개의 열과 233개의 행으로 이루어져 있다.
df.columns.nunique() #15
total_rows = df.shape[0]
total_rows #233
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
plt.show() # 열별 결측치 갯수가 비슷함
## df으로 결측치 직접 확인
df[df.isnull().any(axis=1)][columns_with_missing_values] # 연구 후반 부분의 환경 변수 데이터가 많이 없음
df = df.drop(df.index[199:208]) # 환경 변수와 생리학적 변수의 결측치가 많은 199~208번째 행 제거
df.shape[0] #233->224로 줄어듦
'''
결측치 열 12개
    'mean.Temperature_60',
    'grad.Temperature_60', 
    'mean.Humidity_60', 
    'grad.Humidity_60',
->'Temperature', 'Humidity'ㄴ은 2주 사이에 큰 차이가 없어 평균값으로 결측치 처리
    'mean.Solar_60',
    'grad.Solar_60',
->'Solar'는 시간에 영향을 많이 받기에 'Vote_time'에 맞춰서 결측치 처리
    'mean.hr_60',
    'grad.hr_60',
    'mean.WristT_60',
    'grad.WristT_60',
    'mean.PantT_60',
    'grad.PantT_60',
-> 생리학적 변수 'hr', 'WristT', 'PantT'는 비교적 순간적으로 급격한 변화를 띄지 않는 변수로 주변 10개 평균값으로 결측치 처리
'''
## -> 'Temperature', 'Humidity'은 2주 사이에 큰 차이가 없어 평균값으로 결측치 처리
environmental_features = ['mean.Temperature_60', 'grad.Temperature_60', 'mean.Humidity_60', 'grad.Humidity_60']
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

## -> 생리학적 변수 'hr', 'WristT', 'PantT'는 비교적 순간적으로 급격한 변화를 띄지 않는 변수로 결측치 앞뒤로 20개씩, 총 40개의 평균값으로 결측치 처리(결측치가 다른 subect에 비해 많아서 10->40개로 늘림)
physiological_features = ['mean.hr_60', 'grad.hr_60', 'mean.WristT_60', 'grad.WristT_60','mean.PantT_60', 'grad.PantT_60']
df[physiological_features] = df[physiological_features].fillna(df[physiological_features].rolling(window=41, min_periods=1, center=True).mean())
## missing_values가 있는지 확인
print(df.columns[df.isnull().any()].tolist())
'''
'mean.Solar_60', 
'grad.Solar_60',
'mean.PantT_60', 
'grad.PantT_60', 
'sd.PantT_60', 
'Vote_time_as_number' 
'''
df[['Vote_time', 'Vote_time_as_number']].isnull().sum() # 'Vote_time'에 결측치가 있음을 확인
df['Vote_time_as_number'] = pd.to_datetime(df['Vote_time'], format='%m/%d/%Y %H:%M').dt.hour * 60 + pd.to_datetime(df['Vote_time'], format='%m/%d/%Y %H:%M').dt.minute #다시 계산해서 해결. 주변이 모두 결측치인 행에서 생긴 오류임

df[physiological_features] = df[physiological_features].fillna(df[physiological_features].rolling(window=41, min_periods=1, center=True).mean()) #다시 한번 결측치 처리
## 결측치 확인
print(df.columns[df.isnull().any()].tolist()) #'mean.Solar_60', 'grad.Solar_60', 'sd.Solar_60'에 결측치 있어서 다시 돌림
df[physiological_features] = df[physiological_features].fillna(df[physiological_features].rolling(window=41, min_periods=1, center=True).mean()) #다시 한번 결측치 처리
df[physiological_features] = df[physiological_features].fillna(df[physiological_features].rolling(window=41, min_periods=1, center=True).mean()) #다시 한번 결측치 처리
df[physiological_features] = df[physiological_features].fillna(df[physiological_features].rolling(window=41, min_periods=1, center=True).mean()) #다시 한번 결측치 처리
print(df.columns[df.isnull().any()].tolist()) #결측치가 계속 나옴
df[['mean.Solar_60', 'grad.Solar_60']].isnull().sum() 
'''
계속해서 
mean.Solar_60    9
grad.Solar_60    9
각 열에서 9개의 결측치가 나옴
'''
plt.figure(figsize=(10,10)) #히트맵으로 결측치 시각화
sns.heatmap(df.isnull(), cbar=False) 
plt.show() # 특정행에서만 결측치가 나옴
df[['mean.Solar_60', 'grad.Solar_60']].tail(10) # 뒤에서 9개의 행에서 지속적으로 결측치가 나옴 -> 제거
df = df.drop(df.index[215:]) # 215~마지막번째 행 제거
df.shape[0] # 행의 갯수가 224->215로 줄어듦

## 이상치 확인
df.describe()
df.boxplot(rot=90, figsize=(10,10))
## 환경변수('Temperature', 'Humidity', 'Solar')에 대한 이상치 확인 -> 외부 데이터(기상청)를 가져온 것이므로 이상치 제거하지 않기로 결정
environmental_features_include_solar = ['mean.Temperature_60', 'grad.Temperature_60', 'mean.Humidity_60', 'grad.Humidity_60', 'mean.Solar_60', 'grad.Solar_60']
plt.boxplot(df[environmental_features_include_solar])
## 생리학적 변수 'hr', 'WristT', 'PantT'에 대한 이상치는 순간적인 변화에 의해 생기는 값이므로 개인 therml_sens를 찾는 회귀, 분류 모델이 영향을 많이 줄 것이므로 평균값으로 대체하기로 결정
physiological_features = ['mean.hr_60', 'grad.hr_60', 'mean.WristT_60', 'grad.WristT_60', 'mean.PantT_60', 'grad.PantT_60']
df[physiological_features].boxplot(rot=90, figsize=(10,10))
## Z-score로 이상치 찾고 평균값으로 대체
z = np.abs(stats.zscore(df[physiological_features]))
print(np.where(z > 3)) # 표준정규분포에서 약 99.7%의 데이터가 평균에서 ±3 표준편차 범위에 위치하는 outlier를 이상치로 판단. 'Heart rate' 부분만 조금 처리 하면 되므로 z-score를 3으로 함
df[physiological_features] = df[physiological_features][(z < 3).all(axis=1)] #이상치 제거
df[physiological_features] = df[physiological_features].fillna(df[physiological_features].mean()) #이상치 제거 후 결측치 처리
## 이상치 최종 확인
df[physiological_features].boxplot(rot=90, figsize=(10,10)) #이상치 처리 끝!!!
###############################################################################################
##3-8. 피쳐 스케일링
## 차원축소를 위해 minmaxscaler로 데이터 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[environmental_features_include_solar] = scaler.fit_transform(df[environmental_features_include_solar]) #환경변수 0~1 스케일링
df[physiological_features] = scaler.fit_transform(df[physiological_features]) #생리학적 변수 0~1 스케일링
## 스케일링 확인
df[environmental_features_include_solar].describe()
df[environmental_features_include_solar].boxplot(rot=90, figsize=(10,10))
## Other features(기타 변수) 3개('ID','Vote_time', 'Vote_time_as_number')는 스케일링 하지 않음
## 'therm_sens'는 우리가 구하고자 하는 변수(label_y)이므로 스케일링 하지 않음
###############################################################################################
##3-9. 차원축소
mean_environmental_features = ['mean.Temperature_60', 'mean.Humidity_60', 'mean.Solar_60']
grad_environmental_features = ['grad.Temperature_60', 'grad.Humidity_60', 'grad.Solar_60']
from sklearn.manifold import TSNE
tsne = TSNE(n_components=1, random_state=0)
tsne_mean_environment_features = tsne.fit_transform(df[mean_environmental_features])
tsne_grad_environment_features = tsne.fit_transform(df[grad_environmental_features])
mean_physiological_features = ['mean.hr_60', 'mean.WristT_60', 'mean.PantT_60']
grad_physiological_features = ['grad.hr_60', 'grad.WristT_60', 'grad.PantT_60']
tsne_mean_physiological_features = tsne.fit_transform(df[mean_physiological_features])
tsne_grad_physiological_features = tsne.fit_transform(df[grad_physiological_features])
scaler = MinMaxScaler()
tsne_mean_environment_features = scaler.fit_transform(tsne_mean_environment_features)
tsne_grad_environment_features = scaler.fit_transform(tsne_grad_environment_features)
tsne_mean_physiological_features = scaler.fit_transform(tsne_mean_physiological_features)
tsne_grad_physiological_features = scaler.fit_transform(tsne_grad_physiological_features)
###############################################################################################
##3-10. 데이터 전처리 및 분석 마무리
##차원 축소, 스케일링 된 피쳐들로 데이터 프레임 만들기
df_subject9 = pd.DataFrame({
    'grad_environmental_features': tsne_grad_environment_features.flatten(),
    'mean_physiological_features': tsne_mean_physiological_features.flatten(),
    'grad_physiological_features': tsne_grad_physiological_features.flatten(),
    'therm_sens': df['therm_sens']
})
df_subject9.shape #총 4개의 열과 215개의 행으로 이루어져 있음
###############################################################################################
#4. model 만들기
##4-1. 데이터 분리(train, test='therm_sens')
from sklearn.model_selection import train_test_split
X = df_subject9.iloc[:, :3]
y = df_subject9.iloc[:, 3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
###############################################################################################
##4-2. 5가지 모델 벤치마킹(LinearRegression, SVM, Desicion tree, Random Forest, MLP) + RMSE(이상치에 민감함), MAE(간단함)를 사용해 모델 평가
from sklearn.metrics import mean_squared_error, mean_absolute_error
## LinearRegression을 사용
from sklearn.linear_model import LinearRegression
model_LinearRegression = LinearRegression()
model_LinearRegression.fit(X_train, y_train)
y_pred = model_LinearRegression.predict(X_test)
rmse_model_LinearRegression = np.sqrt(mean_squared_error(y_test, y_pred))
mae_model_LinearRegression = mean_absolute_error(y_test, y_pred)
## SVM을 사용
from sklearn.svm import SVR
model_svm = SVR(kernel='rbf')
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)
rmse_model_svm = np.sqrt(mean_squared_error(y_test, y_pred_svm))
mae_model_svm = mean_absolute_error(y_test, y_pred_svm)
## Decision Tree를 사용
from sklearn.tree import DecisionTreeRegressor
model_DecisionTreeRegressor = DecisionTreeRegressor()
model_DecisionTreeRegressor.fit(X_train, y_train)
y_pred_tree = model_DecisionTreeRegressor.predict(X_test)
rmse_model_DecisionTreeRegressor = np.sqrt(mean_squared_error(y_test, y_pred_tree))
mae_model_DecisionTreeRegressor = mean_absolute_error(y_test, y_pred_tree)
## Random Forest를 사용
from sklearn.ensemble import RandomForestRegressor
model_RandomForestRegressor = RandomForestRegressor()
model_RandomForestRegressor.fit(X_train, y_train)
y_pred_rf = model_RandomForestRegressor.predict(X_test)
rmse_model_RandomForestRegressor = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_model_RandomForestRegressor = mean_absolute_error(y_test, y_pred_rf)
## MLP를 사용
from sklearn.neural_network import MLPRegressor
model_MLPRegressor = MLPRegressor()
model_MLPRegressor.fit(X_train, y_train)
y_pred_mlp = model_MLPRegressor.predict(X_test)
rmse_model_MLPRegressor = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
mae_model_MLPRegressor = mean_absolute_error(y_test, y_pred_mlp)
###############################################################################################
##4-3. 모델 평가 시각화 비교
rmse=[rmse_model_LinearRegression, rmse_model_svm, rmse_model_DecisionTreeRegressor, rmse_model_RandomForestRegressor, rmse_model_MLPRegressor]
mae=[mae_model_LinearRegression, mae_model_svm, mae_model_DecisionTreeRegressor, mae_model_RandomForestRegressor, mae_model_MLPRegressor]
name=['LinearRegression', 'SVM', 'DecisionTreeRegressor', 'RandomForestRegressor', 'MLPRegressor']
y = np.arange(len(mae))
plt.figure(figsize=(10, 6))
plt.barh(y, mae, height=0.4, label='MAE')
plt.barh(y + 0.4, rmse, height=0.4, label='RMSE')
plt.yticks(y + 0.2, name, fontsize=10)
plt.title('Performance Evaluation')
plt.legend()
plt.xlabel('Value')
plt.show()