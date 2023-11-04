#1. 문제 정의 "구성원의 열 쾌적성을 고려한 공간 최적 온도 설정"
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

##3-1. 실내 환경만 남겨두기 Location: (1 = indoor; -1 = outdoor)
df[df["location"]==1] #3297
df[df["location"]==-1] #546

df = df[df["location"]==1]

##3-2. 스마트 워치로 구하기 어려운 피쳐 제거(AnkleT)
df.columns.nunique() #82
columns_to_drop = ['mean.AnkleT_5', 'grad.AnkleT_5', 'sd.AnkleT_5', 'mean.AnkleT_15', 'grad.AnkleT_15', 'sd.AnkleT_15', 'mean.AnkleT_60', 'grad.AnkleT_60', 'sd.AnkleT_60']
df = df.drop(columns=columns_to_drop)
df.columns.nunique() #73

##3-3. 시간별 피쳐 처리


##3-4. 데이터 개인별로 나누기(14명)
for i in df["ID"].unique():
    subject = df[df["ID"]==i]
    subject.to_csv("subject"+str(i)+".csv")
##3-5. 여기부터는 개별 subject.py에서 진행

#######################################################################

### 두 피쳐간 상관도를 보는 코드(숫자 + 그림)
df['mean.hr_60'].corr(df['mean.PantT_60'])
plt.scatter(df['mean.hr_60'], df['mean.AnkleT_60'], alpha=0.7)
plt.show()

### X를 위한 선별된 피처 산점도(객관 데이터)
sns.set(font_scale=1.1) ## 폰트사이즈 조절
sns.set_style('ticks') ## 축 눈금 표시
data = df[[
        'mean.hr_60', 
        'mean.WristT_60',
        'mean.AnkleT_60', 
        'mean.PantT_60']]
sns.pairplot(data, diag_kind=None,  plot_kws={'alpha':0.2})
plt.show()

### X를 위한 선별된 피처 산점도(객관+주관 데이터)
sns.set(font_scale=1.1)# 폰트사이즈 조절
sns.set_style('whitegrid')# 스타일 설정
data = df[[ # 데이터 선택
    'ColdSens', 'ColdExp',
    'mean.hr_60', 
    'mean.WristT_60',
    'mean.AnkleT_60', 
    'mean.PantT_60']]
sns.pairplot(data, diag_kind='kde', markers='o', plot_kws={'alpha':0.4})# Pairplot 생성
plt.suptitle("Pairplot of Weather Variables", y=1.02)# 그래프 제목 추가
plt.show()