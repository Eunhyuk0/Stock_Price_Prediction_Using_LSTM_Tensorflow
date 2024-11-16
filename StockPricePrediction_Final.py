#사용하기 전 preprocessor.py 와 기업 주식 데이터 (XXXX.csv) 를 같은 디렉터리에 넣으세요
#절대적 경로를 설정하는 것이 좋습니다 (e.g. C:/My Files....)
#Before use, put preprocesssor.py and stock data (XXXX.csv) in the same directory.
#absolute path recommended (e.g. C:/My Files....)

import pandas as pd
from datetime import date
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import preprocessor #preprocessor.py
MODEL_PATH = 'keras_model.h5'

#데이터 전처리
StockName = input("type Stock Name (e.g. Apple -> APPL)")
preprocess = input("Preprocess Data? (Y / N)")
if preprocess == 'y' or preprocess == 'Y':
    preprocessor.preprocess(StockName, 0)

scaler = MinMaxScaler(feature_range=(0,1))
name = "YOUR ABSOLUTE PATH"+StockName+"_processed.csv"
data = pd.read_csv(name)

df=data[['Close', 'Volume']]
df.head()
df_scaled = scaler.fit_transform(df)

#80%는 training, 20%는 test 로 사용
train = pd.DataFrame(df_scaled[0:int(len(data)*0.80)])
test = pd.DataFrame(df_scaled[int(len(data)*0.80): int(len(data))])

print(train.shape)
print(test.shape)

data_training_array = train.values

x_train = []
y_train = [] 

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 2)  # Specify 3 features

retrain = input("Retrain Model? (Y / N)")
if retrain == 'n' or retrain == 'N':
    model = load_model(MODEL_PATH)
else:
    model = Sequential([LSTM(units=50, activation = 'tanh', input_shape=(x_train.shape[1], x_train.shape[2])), Dense(units=1)])

    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    model.fit(x_train, y_train,epochs = 35,batch_size = 32)
    model.save('keras_model.h5')

    

past100 = pd.DataFrame(train[-100:])
test_df = pd.DataFrame(test)
final_df = pd.concat([past100, test_df], ignore_index=True) #test_df 에 training data 의 최종 100일 추가 -> final_df

final_df.head()

input_data = scaler.fit_transform(final_df)
input_data.shape

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
   x_test.append(input_data[i-100: i]) 
   y_test.append(input_data[i, 0]) 

x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)

#예측 값 생성
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mae_percentage = (mae / np.mean(y_test)) * 100
print("Mean absolute error on test set: {:.2f}%".format(mae_percentage))

plt.switch_backend('TkAgg')
plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = "Original Price")
plt.plot(y_pred, 'r', label = "Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()