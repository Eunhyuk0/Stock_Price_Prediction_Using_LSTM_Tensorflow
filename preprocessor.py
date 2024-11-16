import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MinMax = {} #데이터 복원을 위해 최소, 최댓값 저장

#데이터를 0~1 값으로 맞추는 함수
def ScaleData(DATA):
    global MinMax
    for column in DATA.select_dtypes(include=['float64', 'int64']):
        min_value = DATA[column].min()
        max_value = DATA[column].max()
        MinMax[column] = (min_value, max_value)
        DATA[column] = (DATA[column] - min_value) / (max_value - min_value)
    return DATA

#2가지 data 를 출력해 변화를 볼 수 있는 함수
def PrintAB_Datas(a, b):
    plt.figure(figsize=(14, 10))

    # a 의 Close Price
    plt.subplot(2, 2, 1)
    plt.scatter(a.index, a['Close'], color='blue', label='Original Close Price', s=10)
    plt.title('Raw Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()

    # a 의 Volume
    plt.subplot(2, 2, 2)
    plt.scatter(a.index, a['Volume'], color='orange', label='Original Volume', s=10)
    plt.title('Raw Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()

    # b 의 Close Price
    plt.subplot(2, 2, 3)
    plt.scatter(b.index, b['Close'], color='green', label='Preprocessed Close Price', s=10)
    plt.title('Preprocessed Close Price')
    plt.xlabel('Date')
    plt.ylabel('Normalized Close Price')
    plt.legend()

    # b 의 Volume
    plt.subplot(2, 2, 4)
    plt.scatter(b.index, b['Volume'], color='red', label='Preprocessed Volume', s=10)
    plt.title('Preprocessed Volume')
    plt.xlabel('Date')
    plt.ylabel('Normalized Volume')
    plt.legend()

    plt.tight_layout()
    plt.show()

#전처리 작업을 수행하는 함수, main 에서 import 해서 사용하기 위함
def preprocess(filename, printdata):
    Thresh_StdDev = 3.5 # 표준편차로 특이한 값 제거할 때 쓰는 기준 (n배 이상...)
    Thresh_Diff = 0.19 # 앞뒤 2개와 값 차이로 값 제거할 때 쓰는 기준 (차이가 n)
    MaxValue = 1 #최댓값
    minValue = 0 #최솟값
    #전처리하는 데이터 (csv)
    name = "YOUR ABSOLUTE PATH"+filename+".csv"
    outputname = "YOUR ABSOLUTE PATH"+filename+"_processed.csv"
    data = pd.read_csv(name)

    #원본 저장
    data_raw = data.copy()
    data_raw_scaled = ScaleData(data_raw)

    # 'Date' 를 pandas 의 datetime 규격으로 맞춤
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data_timescaled = data.copy()

    #Standard Deviation (표준편차) 의 Threshold배보다 크거나 작은 값 제거
    StdDeviation = (((data['Volume'] - data['Volume'].mean()) ** 2).sum() / len(data)) ** 0.5
    data = data[(data['Volume'] >= data['Volume'].mean() - Thresh_StdDev * StdDeviation) & 
                (data['Volume'] <= data['Volume'].mean() + Thresh_StdDev * StdDeviation)]

    data = data.dropna() #pandas 를 이용해 값이 제거된 빈 줄 삭제
    #1차 전처리된 데이터 저장
    data_preprocessed = data.copy()

    data = ScaleData(data)
    #0~1에 맞춘 데이터 저장
    data_scaled = data.copy()

    Deleting = []
    for i in range(len(data)):
        #주변 값과의 차이가 일정 수치 이상이면 값 제거
        if i == 0:
            if abs(data['Volume'].iloc[i] - data['Volume'].iloc[i + 1]) > Thresh_Diff:
                data['Volume'][i] = np.nan
        elif i == len(data) - 1:
            if abs(data['Volume'].iloc[i] - data['Volume'].iloc[i - 1]) > Thresh_Diff:
                data['Volume'][i] = np.nan
        else:
            if (abs(data['Volume'].iloc[i] - data['Volume'].iloc[i - 1]) > Thresh_Diff) or (abs(data['Volume'].iloc[i] - data['Volume'].iloc[i + 1]) > Thresh_Diff):
                data['Volume'][i] = np.nan

    data = data.dropna() #pandas 를 이용해 값이 제거된 빈 줄 삭제

    for col in data:
        for i in range(len(data)):        
            #과도하게 크거나 작은 값을 설정한 최소,최대 값으로 변경
            if data[col].iloc[i] > MaxValue:
                data.at[i, col] = MaxValue
            if data[col].iloc[i] <= minValue:
                data.at[i, col] = minValue

    data = data.dropna() #pandas 를 이용해 값이 제거된 빈 줄 삭제       

    # Drop rows where any column contains '1' or '0'
    data = data[~((data == 1) | (data == 0)).any(axis=1)]

    if(printdata):
        #원하는 값으로 출력
        PrintAB_Datas(data_timescaled, data)

    # 저장
    data.to_csv(outputname, index=True)

