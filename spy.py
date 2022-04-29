# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:25:24 2022

@author: ACER
"""

# import yfinance as yf

import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

app = Flask(__name__)
#See the yahoo finance ticker for your stock symbol
key = "757f6d561ad6c59ce1d9b6d4ca73328956fdea48"
stock_symbol = 'GAIL.NS'

@app.route('/', methods=['POST'])
def json():
    request_data = request.get_json()
    company = request_data['comp']
    n_epoch = request_data['epoch']
    n_days= request_data['days']

    df = pdr.get_data_tiingo(company, api_key=key)
    df.to_csv(f'dataset/{company}.csv')
    data=pd.read_csv(f'dataset/{company}.csv')
    #last 5 years data with interval of 1 day
    # data = yf.download(tickers=stock_symbol,period='5y',interval='1d')

    print("type",type(data))

    print("data.head()",data.head())

    print("len(data)",len(data))

    print("data.tail()",data.tail())

    opn = data[['open']]

    opn.plot()

    ds = opn.values
    ds
    plt.plot(ds)
    plt.show()

    #Using MinMaxScaler for normalizing data between 0 & 1
    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

    len(ds_scaled), len(ds)

    #Defining test and train data sizes
    train_size = int(len(ds_scaled)*0.70)
    test_size = len(ds_scaled) - train_size

    train_size,test_size

    #Splitting data between train and test
    ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

    len(ds_train),len(ds_test)

    #creating dataset in time series for LSTM model 
    #X[100,120,140,160,180] : Y[200]
    def create_ds(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)

    #Taking 100 days price as one record for training
    time_stamp = 100
    X_train, y_train = create_ds(ds_train,time_stamp)
    X_test, y_test = create_ds(ds_test,time_stamp)

    X_train.shape,y_train.shape

    X_test.shape, y_test.shape

    #Reshaping data to fit into LSTM model
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    #Creating LSTM model using keras
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1,activation='linear'))
    model.summary()

    #Training model with adam optimizer and mean squared error loss function
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=1,batch_size=64)

    #PLotting loss, it shows that loss has decreased significantly and model trained well
    loss = model.history.history['loss']
    plt.plot(loss)
    plt.show()

    #Predicitng on train and test data
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    #Inverse transform to get actual value
    train_predict = normalizer.inverse_transform(train_predict)
    test_predict = normalizer.inverse_transform(test_predict)

    #Comparing using visuals
    plt.plot(normalizer.inverse_transform(ds_scaled))
    plt.plot(train_predict)
    plt.plot(test_predict)
    plt.show()

    type(train_predict)
    test = np.vstack((train_predict,test_predict))

    #Combining the predited data to create uniform data visualization
    plt.plot(normalizer.inverse_transform(ds_scaled))
    plt.plot(test)
    plt.show()

    len(ds_test)
    print("len(ds_test)",len(ds_test))

    #Getting the last 100 days records
    fut_inp = ds_test[(len(ds_test)-100):]

    fut_inp = fut_inp.reshape(1,-1)
    tmp_inp = list(fut_inp)
    fut_inp.shape

    #Creating list of the last 100 data
    tmp_inp = tmp_inp[0].tolist()

    #Predicting next 30 days price suing the current data
    #It will predict in sliding window manner (algorithm) with stride 1
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        
        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
        

    print(lst_output)

    len(ds_scaled)

    print("len(ds_scaled)",len(ds_scaled))

    #Creating a dummy plane to plot graph one after another
    plot_new=np.arange(1,101)
    plot_pred=np.arange(101,131)

    plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[len(ds_scaled)-100:]))
    plt.plot(plot_pred, normalizer.inverse_transform(lst_output))
    plt.show()

    ds_new = ds_scaled.tolist()
    len(ds_new)

    #Entends helps us to fill the missing value with approx value
    ds_new.extend(lst_output)
    plt.plot(ds_new[1200:])
    plt.show()

    #Creating final data for plotting
    final_graph = normalizer.inverse_transform(ds_new).tolist()

    #Plotting final results with predicted value after 30 Days
    plt.plot(final_graph,)
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.title("{0} prediction of next month open".format(company))
    plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
    plt.legend()
    plt.show()

    return jsonify(
        costForfuture=format(round(float(*final_graph[len(final_graph)-1]),2))
    )

if __name__ == '__main__':
    app.run(debug=True)
