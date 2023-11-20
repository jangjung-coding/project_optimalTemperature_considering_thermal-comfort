#1. 문제 정의 "구성원의 열 쾌적성을 고려한 공간 최적 온도 설정 ft.스마트 워치"
#2. 데이터 수집 "https://view.officeapps.live.com/op/view.aspx?src=https://raw.githubusercontent.com/jangjung-coding/lesson/master/Description%20of%20variables_Liu.docx&wdOrigin=BROWSELINK" + 직접 측정
#3. 데이터 전처리 및 분석
'''Units
Temperature: ˚C
Wind speed: m/s
Heart rate: bpm
Acceleration: m/s2
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = 'raw_data_Liu.csv' #전체 데이터 경로
df= pd.read_csv(path) 

##3-1. 실내 환경만 남기고 열 제거 Location: (1 = indoor; -1 = outdoor)
df[df["location"]==1] #3297 indoor data
df[df["location"]==-1] #546 outdoor data

df = df[df["location"]==1]
df = df.drop(columns=['location'])
## windVelocity는 실내이므로 제거
df = df.drop(columns=[ 'mean.Winvel_60', 'grad.Winvel_60', 'sd.Winvel_60', 'mean.Winvel_480', 'grad.Winvel_480', 'sd.Winvel_480'])
###############################################################################################
##3-2. 스마트 워치로 구하기 어려운 피쳐 제거(AnkleT. 발목온도는 스마트 워치로 측정하기 어려움)
df.columns.nunique() #75
columns_to_drop = ['mean.AnkleT_5', 'grad.AnkleT_5', 'sd.AnkleT_5', 'mean.AnkleT_15', 'grad.AnkleT_15', 'sd.AnkleT_15', 'mean.AnkleT_60', 'grad.AnkleT_60', 'sd.AnkleT_60']
df = df.drop(columns=columns_to_drop)
df.columns.nunique() #66
###############################################################################################
##3-3. 시간별 피쳐 처리
'''
- Environmental features(환경 변수)
       'mean.Temperature_60', 'grad.Temperature_60', 'sd.Temperature_60',
       'mean.Temperature_480', 'grad.Temperature_480', 'sd.Temperature_480',
       'mean.Humidity_60', 'grad.Humidity_60', 'sd.Humidity_60',
       'mean.Humidity_480', 'grad.Humidity_480', 'sd.Humidity_480',
       'mean.Solar_60', 'grad.Solar_60', 'sd.Solar_60', 
       'mean.Solar_480', 'grad.Solar_480', 'sd.Solar_480',
- Physiological features(생리학적 변수)
       'mean.hr_5', 'grad.hr_5', 'sd.hr_5', 
       'mean.hr_15', 'grad.hr_15', 'sd.hr_15', 
       'mean.hr_60', 'grad.hr_60', 'sd.hr_60', 
       'mean.WristT_5', 'grad.WristT_5', 'sd.WristT_5', 
       'mean.WristT_15', 'grad.WristT_15', 'sd.WristT_15', 
       'mean.WristT_60', 'grad.WristT_60', 'sd.WristT_60',
       'mean.PantT_5', 'grad.PantT_5', 'sd.PantT_5', 
       'mean.PantT_15', 'grad.PantT_15', 'sd.PantT_15', 
       'mean.PantT_60', 'grad.PantT_60', 'sd.PantT_60', 
       'mean.act_5', 'grad.act_5', 'sd.act_5', 
       'mean.act_15', 'grad.act_15', 'sd.act_15', 
       'mean.act_60', 'grad.act_60', 'sd.act_60'],
'''
## meanTenperature 시간별 비교
plt.plot(df['mean.Temperature_60'],alpha=0.2, color='red') 
plt.plot(df['mean.Temperature_480'], alpha=0.2, color='blue')
df['mean.Temperature_60'].corr(df['mean.Temperature_480']) #0.99 - 60분과 480분 데이터는 거의 비슷하다
## meanHumidity 시간별 비교
plt.plot(df['mean.Humidity_60'],alpha=0.2, color='red')
plt.plot(df['mean.Humidity_480'], alpha=0.2, color='blue')
df['mean.Humidity_60'].corr(df['mean.Humidity_480']) #0.99 - 60분과 480분 데이터는 거의 비슷하다
## meanWinvel 시간별 비교
plt.plot(df['mean.Winvel_60'],alpha=0.2, color='red')
plt.plot(df['mean.Winvel_480'], alpha=0.2, color='blue')
df['mean.Winvel_60'].corr(df['mean.Winvel_480']) #0.98 - 60분과 480분 데이터는 거의 비슷하다
## menaSolar 시간별 비교
plt.plot(df['mean.Solar_60'],alpha=0.2, color='red')
plt.plot(df['mean.Solar_480'], alpha=0.2, color='blue')
df['mean.Solar_60'].corr(df['mean.Solar_480']) #0.98 - 60분과 480분 데이터는 거의 비슷하다
## mean.hr 시간별 비교
plt.plot(df['mean.hr_5'],alpha=0.2, color='red')
plt.plot(df['mean.hr_15'], alpha=0.2, color='blue')
plt.plot(df['mean.hr_60'], alpha=0.2, color='black')
df['mean.hr_5'].corr(df['mean.hr_15']) #0.94
df['mean.hr_5'].corr(df['mean.hr_60']) #0.87
df['mean.hr_15'].corr(df['mean.hr_60']) #0.93
## mean.WristT 시간별 비교
plt.plot(df['mean.WristT_5'],alpha=0.2, color='red')
plt.plot(df['mean.WristT_15'], alpha=0.2, color='blue')
plt.plot(df['mean.WristT_60'], alpha=0.2, color='black')
df['mean.WristT_5'].corr(df['mean.WristT_15']) #0.95
df['mean.WristT_5'].corr(df['mean.WristT_60']) #0.84
df['mean.WristT_15'].corr(df['mean.WristT_60']) #0.91
## mean.PantT 시간별 비교
plt.plot(df['mean.PantT_5'],alpha=0.2, color='red')
plt.plot(df['mean.PantT_15'], alpha=0.2, color='blue')
plt.plot(df['mean.PantT_60'], alpha=0.2, color='black')
df['mean.PantT_5'].corr(df['mean.PantT_15']) #0.97
df['mean.PantT_5'].corr(df['mean.PantT_60']) #0.90
df['mean.PantT_15'].corr(df['mean.PantT_60']) #0.94
## mean.act 시간별 비교 -> 다른 feature들과 다르게 시간별 차이가 크다 
plt.plot(df['mean.act_5'],alpha=0.2, color='red')
plt.plot(df['mean.act_15'], alpha=0.2, color='blue')
plt.plot(df['mean.act_60'], alpha=0.2, color='black')
df['mean.act_5'].corr(df['mean.act_15']) #0.87
df['mean.act_5'].corr(df['mean.act_60']) #0.64
df['mean.act_15'].corr(df['mean.act_60']) #0.72
## -> 손목의 가속도를 나타내는 act feature 규칙이 없고 시간에 따라 변화가 크다
plt.plot(df['grad.act_5'],alpha=0.2, color='red')
plt.plot(df['grad.act_15'], alpha=0.2, color='blue')
plt.plot(df['grad.act_60'], alpha=0.2, color='black')
df['grad.act_5'].corr(df['grad.act_15']) #0.70
df['grad.act_5'].corr(df['grad.act_60']) #0.30
df['grad.act_15'].corr(df['grad.act_60']) #0.41
## -> 손목의 가속도를 나타내는 act feature는 therm_sens와 상관관계가 없다
plt.plot(df['mean.act_5'],alpha=0.2, color='red')
plt.plot(df['therm_sens'], alpha=0.2, color='blue')
df['mean.act_5'].corr(df['therm_sens']) #-0.01
df['mean.act_15'].corr(df['therm_sens']) #0.00
df['mean.act_60'].corr(df['therm_sens']) #0.00
## act feature는 시간별 관계가 떨어지고, therm_sens와도 상관관계가 없고, 손목 움직임이 신체 열을 대표하지 못하는 정적인 사무실 구성원의 특성을 고려하면 제거하는 것이 좋다고 판단
df = df.drop(columns=['mean.act_5', 'grad.act_5', 'sd.act_5', 'mean.act_15', 'grad.act_15', 'sd.act_15', 'mean.act_60', 'grad.act_60', 'sd.act_60'])
df.columns.nunique() #57
## 나머지 feature들은 mean, grad, sd 모두 시간별로 비슷한 경향을 보임. 또한 스마트 워치로 온도설정을 하는 주기를 한시간으로 잡아서 mean, grad, sd.60분 데이터만 사용하기로 결정.
df = df.drop(columns=[
       'mean.Temperature_480', 'grad.Temperature_480', 'sd.Temperature_480',
       'mean.Humidity_480', 'grad.Humidity_480', 'sd.Humidity_480', 
       'mean.Solar_480', 'grad.Solar_480', 'sd.Solar_480',
       'mean.hr_5', 'grad.hr_5', 'sd.hr_5', 'mean.hr_15', 'grad.hr_15', 'sd.hr_15',
       'mean.WristT_5', 'grad.WristT_5', 'sd.WristT_5', 'mean.WristT_15', 'grad.WristT_15', 'sd.WristT_15',
       'mean.PantT_5', 'grad.PantT_5', 'sd.PantT_5', 'mean.PantT_15', 'grad.PantT_15', 'sd.PantT_15'])
df.columns.nunique() #30
###############################################################################################
##3-4. 'therm_sens'와 'therm_pref'는 비슷한 의미로 모두 thermal comfort를 나타내는 변수이므로 7가지로 보다 자세한 지표인 'therm_sens'만 사용하기로 결정
df = df.drop(columns=['therm_pref'])
df.columns.nunique() #29
###############################################################################################
##3-5. 데이터 subject별 분리 전 thermal_sens를 예측하는데 영향을 미치지 않는 피쳐 제거 
df['Coffeeintake'].corr(df['therm_sens']) #0.07
df['Workhr'].corr(df['therm_sens']) #0.13
plt.plot(df[df['ID']==7]['Workhr']) #2주 동안 개인별 운동시간은 일정했고 계속해서 변하는 thermal_sens를 구하는데 있어 고정적인 특성의 운동 시간은 관계가 없다고 판단
## -> 'Coffeeintake'와 'therm_sens'의 상관관계는 0.07로 매우 낮다. 또한 정확한 섭취 시간을 알 수 없어 차원의 저주를 고려해 제거하기로 결정
## -> 'Workhr'와 'therm_sens'의 상관관계는 0.13으로 매우 낮다. 또한 정확한 운동 시간을 알 수 없어 차원의 저주를 고려해 제거하기로 결정
df = df.drop(columns=['Coffeeintake', 'Workhr'])
##sd 값들을 보면 변동성이 적고 피쳐 특성상 'therma_sens'를 예측하는데 의미가 없어서 제외
plt.plot(df['sd.Temperature_60'], alpha=0.2, color='blue')
df['sd.Temperature_60'].corr(df['therm_sens']) #0.04
df = df.drop(columns=['sd.Temperature_60', 'sd.Humidity_60', 'sd.Solar_60','sd.hr_60', 'sd.WristT_60', 'sd.PantT_60'])
df.columns.nunique() #21

plt.scatter(df['Sex'], df['therm_sens'])
plt.scatter(df['Age'], df['therm_sens'])
plt.scatter(df['Height'], df['therm_sens'])
plt.scatter(df['Weight'], df['therm_sens'])
## -> Antropometric features(인체 측정 변수)는 단기간인 2주 동안 변하지 않고 'therma_sens'를 예측하는데 높은 상관성을 보이지 않고 개인 데이터만 학습하는 스마트 워치의 특성상 외부 데이터를 사용해서 비교를 할 수 없으므로 제거. 후에 온라인 데이터로 상대적인 비교를 하며 향상시킬 수 있다면 추가할 예정
df = df.drop(columns=['Sex', 'Age', 'Height', 'Weight'])

plt.plot(df['ColdSens'])
plt.plot(df['ColdExp'])
plt.scatter(df['ColdSens'], df['ColdExp'], alpha=0.7)
## -> 'ColdSens'와 'ColdExp'는 비슷한 의미로 모두 개인의 추위에 대한 민감도를 나타내는 변수이다. 하지만 논문에 따라 개인 열 쾌적성을 잘 나타내지 못했고 변하지 않는 값이라는 특성으로 모델에 영향을 줄 수 없으므로 제거
df = df.drop(columns=['ColdSens', 'ColdExp'])
df.columns.nunique() #15
'''
지금까지 남은 15개의 피쳐는 다음과 같다.
- Physiological features(생리학적 변수) 6개
        'mean.hr_60', 'grad.hr_60'
        'mean.WristT_60', 'grad.WristT_60'
        'mean.PantT_60', 'grad.PantT_60'
- Environmental features(환경 변수)  6개
        'mean.Temperature_60', 'grad.Temperature_60'
        'mean.Humidity_60', 'grad.Humidity_60'
        'mean.Solar_60', 'grad.Solar_60'
- Comfortability features(쾌적성 변수) 1개
        'therm_sens'
- Other features(기타 변수) 2개
        'ID', 'Vote_time'
'''
###############################################################################################
##3-6. 데이터 개인별로 나누기(14명)
for i in df["ID"].unique():
    subject = df[df["ID"]==i]
    subject.to_csv("subject"+str(i)+".csv")
    
##3-6. 여기부터는 개별 subject.py에서 진행(이상치, 결측치 처리, 모델링, 성능평가, 적용)
##subject별 결측치와 이상치가 다를것이므로 데이터를 최대한 살리기 위해 각각의 subject.py에서 처리