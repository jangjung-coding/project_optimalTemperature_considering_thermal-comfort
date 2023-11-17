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

#subject1은 15개의 열과 146개의 행으로 이루어져 있다.
df.columns.nunique() #15
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
결측치 열 12개
    'mean.Temperature_60',
    'grad.Temperature_60', 
    'mean.Humidity_60', 
    'grad.Humidity_60',
->'Temperature', 'Humidity', 'Winvel'은 2주 사이에 큰 차이가 없어 평균값으로 결측치 처리
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
## -> 생리학적 변수 'hr', 'WristT', 'PantT'는 비교적 순간적으로 급격한 변화를 띄지 않는 변수로 결측치 앞뒤로 5개씩, 총 10개의 평균값으로 결측치 처리
physiological_features = ['mean.hr_60', 'grad.hr_60', 'mean.WristT_60', 'grad.WristT_60','mean.PantT_60', 'grad.PantT_60']
df[physiological_features] = df[physiological_features].fillna(df[physiological_features].rolling(window=11, min_periods=1, center=True).mean())
## missing_values가 있는지 확인
print(df.columns[df.isnull().any()].tolist())
'''
'mean.WristT_60',
'grad.WristT_60',
'mean.PantT_60',
'grad.PantT_60'
'''
df[physiological_features] = df[physiological_features].fillna(df[physiological_features].rolling(window=11, min_periods=1, center=True).mean()) #다시 한번 결측치 처리
## 결측치 최종 확인
print(df.columns[df.isnull().any()].tolist()) #결측치 없음!!!

## 이상치 확인
df.describe()
df.boxplot(rot=90, figsize=(10,10))
## 환경변수('Temperature', 'Humidity', 'Solar')에 대한 이상치 확인 -> 외부 데이터(기상청)를 가져온 것이므로 이상치 제거하지 않기로 결정
environmental_features_include_solar = ['mean.Temperature_60', 'grad.Temperature_60', 'mean.Humidity_60', 'grad.Humidity_60', 'mean.Solar_60', 'grad.Solar_60']
plt.boxplot(df[environmental_features_include_solar]) # 'Solar'에서 이상치가 많이 발견됨. 이는 subject1 측정 계절이  12월, 즉 겨울 샌프란시스코 일간 온도차가 크기 때문으로 판단
## 생리학적 변수 'hr', 'WristT', 'PantT'에 대한 이상치는 순간적인 변화에 의해 생기는 값이므로 개인 therml_sens를 찾는 회귀, 분류 모델이 영향을 많이 줄 것이므로 평균값으로 대체하기로 결정
physiological_features = ['mean.hr_60', 'grad.hr_60', 'mean.WristT_60', 'grad.WristT_60', 'mean.PantT_60', 'grad.PantT_60']
df[physiological_features].boxplot(rot=90, figsize=(10,10))
## Z-score로 이상치 찾고 평균값으로 대체
z = np.abs(stats.zscore(df[physiological_features]))
print(np.where(z > 3)) # 표준정규분포에서 약 99.7%의 데이터가 평균에서 ±3 표준편차 범위에 위치하는 outlier를 이상치로 판단
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
## 스케일링 된 환경변수의 시각화로 분포 확인
for feature in environmental_features_include_solar:
    sns.kdeplot(df[feature], label=feature, fill=True, linewidth=2)
plt.title('Distribution of Scaled Environmental Features')
plt.xlabel('Scaled Value')
plt.ylabel('Density')
plt.legend(fontsize='small')
plt.show()

df[physiological_features].describe()
df[physiological_features].boxplot(rot=90, figsize=(10,10))
## 스케일링 된 생리학적 변수의 시각화로 분포 확인
for feature in physiological_features:
    sns.kdeplot(df[feature], label=feature, fill=True, linewidth=2)
plt.title('Distribution of Scaled Environmental Features')
plt.xlabel('Scaled Value')
plt.ylabel('Density')
plt.legend(fontsize='small')
plt.show()
## Other features(기타 변수) 3개('ID','Vote_time', 'Vote_time_as_number')는 스케일링 하지 않음
## 'therm_sens'는 우리가 구하고자 하는 변수(label_y)이므로 스케일링 하지 않음
###############################################################################################
##3-9. 차원 축소
## 'therm_sens'를 잘 예측하고 차원의 저주를 피하기 위해 비슷한 성격의 변수들로 새로운 변수를 만들어 차원 축소를 진행하기로 결정
'''
- Environmental features(환경 변수) 6개 -> 2개
        'mean.Temperature_60', 'grad.Temperature_60'
        'mean.Humidity_60', 'grad.Humidity_60'
        'mean.Solar_60', 'grad.Solar_60'
- Physiological features(생리학적 변수) 6개 -> 2개
        'mean.hr_60', 'grad.hr_60'
        'mean.WristT_60', 'grad.WristT_60'
        'mean.PantT_60', 'grad.PantT_60'
- Comfortability features(쾌적성 변수) 1개(우리가 구하고자 하는 변수. y_label)
        'therm_sens'
- Other features(기타 변수) 3개 -> 0개
        'ID', 'Vote_time', 'Vote_time_as_number' 
-> 총 X 피쳐 갯수 5개, 총 Y 피쳐 갯수 1개
'''
## Environmental features(환경 변수) 6개
pd.plotting.scatter_matrix(df[environmental_features_include_solar], alpha=0.8, figsize=(15, 15), diagonal='kde')
## Temperature, Humidity, Solar의 3개 변수에서 mean, grad로 새로운 변수를 만들어 관계 파악
mean_environmental_features = ['mean.Temperature_60', 'mean.Humidity_60', 'mean.Solar_60']
grad_environmental_features = ['grad.Temperature_60', 'grad.Humidity_60', 'grad.Solar_60']
## mean_environmental_features 상관관계 시각화
pd.plotting.scatter_matrix(df[mean_environmental_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
pd.plotting.scatter_matrix(df[grad_environmental_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
## -> mean, grad, sd으로 나누니까 선형적인 상관관계가 아닌 군집화된 데이터 특성이 보임
Temperature_features = ['mean.Temperature_60', 'grad.Temperature_60']
Humidity_features = ['mean.Humidity_60', 'grad.Humidity_60']
Solar_features = ['mean.Solar_60', 'grad.Solar_60']
pd.plotting.scatter_matrix(df[Temperature_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
pd.plotting.scatter_matrix(df[Humidity_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
pd.plotting.scatter_matrix(df[Solar_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
## -> Temperature, Humidity, Solar로 나누니까 mean, grad로 나눈 것보다 비선형적이고 군집화가 적어 특별한 규칙이 보이지 않음
## 따라서 환경변수 6개를 각각의 mean, grad로 나누어 차원 축소를 진행하기로 결정 (6개 -> 2개)
## tsne(t-Distributed Stochastic Neighbor Embedding)를 이용한 환경 변수 차원 축소
from sklearn.manifold import TSNE
tsne = TSNE(n_components=1, random_state=0)
tsne_mean_environment_features = tsne.fit_transform(df[mean_environmental_features])
tsne_grad_environment_features = tsne.fit_transform(df[grad_environmental_features])
## Physiological features(생리학적 변수) 6개
pd.plotting.scatter_matrix(df[physiological_features], alpha=0.8, figsize=(15, 15), diagonal='kde')
mean_physiological_features = ['mean.hr_60', 'mean.WristT_60', 'mean.PantT_60']
grad_physiological_features = ['grad.hr_60', 'grad.WristT_60', 'grad.PantT_60']
pd.plotting.scatter_matrix(df[mean_physiological_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
pd.plotting.scatter_matrix(df[grad_physiological_features], alpha=0.8, figsize=(12, 12), diagonal='kde')
## -> 생리학적 변수도 환경변수와 같은 성격을 띄어 mean, grad로 나누어 차원 축소를 진행하기로 결정 (6개 -> 2개)
## tsne(t-Distributed Stochastic Neighbor Embedding)를 이용한 생리학적 변수 차원 축소
tsne_mean_physiological_features = tsne.fit_transform(df[mean_physiological_features])
tsne_grad_physiological_features = tsne.fit_transform(df[grad_physiological_features])
## 편향되지 않은 모델 학습을 위해 tsne로 얻은 6개 피쳐들 minmaxscaler로 스케일링
scaler = MinMaxScaler()
tsne_mean_environment_features = scaler.fit_transform(tsne_mean_environment_features)
tsne_grad_environment_features = scaler.fit_transform(tsne_grad_environment_features)
tsne_mean_physiological_features = scaler.fit_transform(tsne_mean_physiological_features)
tsne_grad_physiological_features = scaler.fit_transform(tsne_grad_physiological_features)
## ID와 측정 시간을 나타내는 'Vote_time'은 string 타입이고 'Vote_time_as_number'는 Solar결측치를 처리하기 위해 만든 임시 변수라 제거 
'''
최종 선별 피쳐들 5개
- Environmental features(환경 변수) 2개
        'mean_environmental_features', 'grad_environmental_features'
- Physiological features(생리학적 변수) 2개
        'mean_physiological_features', 'grad_physiological_features'
- Comfortability features(쾌적성 변수) 1개(우리가 구하고자 하는 변수. y_label)
        'therm_sens'
-> 피쳐 갯수 4개로 'therm_sens'를 예측하기 위한 데이터 프레임 완성
'''
###############################################################################################
##3-10. 데이터 전처리 및 분석 마무리
##차원 축소, 스케일링 된 피쳐들로 데이터 프레임 만들기
df_subject1 = pd.DataFrame({
    'mean_environmental_features': tsne_mean_environment_features.flatten(),
    'grad_environmental_features': tsne_grad_environment_features.flatten(),
    'mean_physiological_features': tsne_mean_physiological_features.flatten(),
    'grad_physiological_features': tsne_grad_physiological_features.flatten(),
    'therm_sens': df['therm_sens']
})
df_subject1.shape #총 5개의 열과 146개의 행으로 이루어져 있음
###############################################################################################
#4. model 만들기
##4-1. 데이터 분리(train, test='therm_sens')
from sklearn.model_selection import train_test_split
X = df_subject1.iloc[:, :4]
y = df_subject1.iloc[:, 4]
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
###############################################################################################
##4-4. 모델 선정
## SHAP을 통해 피쳐를 바꾼 여러 환경에서 모두 성능이 우수하게 나온 Random Forest를 선정
import shap 
explainer = shap.TreeExplainer(model_LinearRegression)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

df_subject1_1 = pd.DataFrame({
    'grad_environmental_features': tsne_grad_environment_features.flatten(),
    'mean_physiological_features': tsne_mean_physiological_features.flatten(),
    'grad_physiological_features': tsne_grad_physiological_features.flatten(),
    'therm_sens': df['therm_sens']
})
X = df_subject1_1.iloc[:, :3]
y = df_subject1_1.iloc[:, 3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

df_subject1_2 = pd.DataFrame({
    'mean_environmental_features': tsne_grad_environment_features.flatten(),
    'mean_physiological_features': tsne_mean_physiological_features.flatten(),
    'grad_physiological_features': tsne_grad_physiological_features.flatten(),
    'therm_sens': df['therm_sens']
})
X = df_subject1_2.iloc[:, :3]
y = df_subject1_2.iloc[:, 3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

df_subject1_3 = pd.DataFrame({
    'mean_physiological_features': tsne_mean_physiological_features.flatten(),
    'grad_physiological_features': tsne_grad_physiological_features.flatten(),
    'therm_sens': df['therm_sens']
})
X = df_subject1_3.iloc[:, :2]
y = df_subject1_3.iloc[:, 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
## -> df_subject1_1환경(학습 변수 3개)에서 Random Forest모델이 가장 성능이 좋음!!! 




