import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import MinMaxScaler
import keras
import pickle


def staticForecast(serverID, period, startDate, endDate) -> dict:
    """
    在数据集较小的情况下，每次请求将训练全部数据的模型
    :param serverID: 服务器 ID
    :param period: 周期
    :param startDate: 开始日期
    :param endDate: 结束日期
    :return: {list_used : [], list_predict : []}
    """
    # 读取对应节点数据
    data = pd.read_csv(f"datasets/{serverID}_.csv", index_col='collect_time', parse_dates=['collect_time'])
    standard = data['used_value'].max() * 1.01

    # 训练模型
    model = AutoReg(data, lags=40)
    results = model.fit()

    # 根据日期筛选数据
    startDate = pd.Timestamp(startDate)
    endDate = pd.Timestamp(endDate)

    # 按周期生成历史数据
    list_used = []
    tempDate = startDate
    while tempDate < data.index[-1]:
        list_used.append({'collect_time': tempDate, 'timestamp': tempDate.timestamp(),
                          'used_value': data.loc[tempDate][0]})
        tempDate = tempDate + pd.Timedelta(days=period)

    # 生成预测值
    list_predict = []
    if endDate > data.index[-1]:
        predictions = results.predict(start=data.index[-1], end=endDate)
        while tempDate < endDate:
            list_predict.append({'predict_time': tempDate, 'timestamp': tempDate.timestamp(),
                                 'predict_value': predictions.loc[tempDate],
                                 'isAlarm': bool(predictions.loc[tempDate] > standard)})
            tempDate = tempDate + pd.Timedelta(days=period)

    return {'list_used': list_used, 'list_predict': list_predict}


def dynamicForecast(serverID, startTime, endTime) -> dict:
    """
    动态阈值预测算法
    :param serverID: 服务器节点 ID
    :param startTime: 开始时间
    :param endTime: 结束时间
    :return: {list_used : [], list_predict : []}
    """
    # 读取数据，索引为日期，值为容量
    data = pd.read_csv(f'datasets/{serverID}.csv', index_col='date', parse_dates=True).drop(columns='timestamp')
    # 数据缩放器
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_data = scaler.fit_transform(data)
    # 预测周期，两分钟一个数据，一天有 720个数据
    period = 720
    # 读取预训练好的 LSTM 模型
    model = keras.models.load_model(f"datasets/{serverID}.h5")
    # 可以生成预测数据的时间区间
    startTime = max([pd.Timestamp(startTime), data.index[0] + pd.Timedelta(days=1)])
    # if startTime.minute % 2 == 0:
    #     startTime = startTime + pd.Timedelta(minutes=1)
    if startTime.minute % 2 != data.index[0].minute % 2:
        startTime = startTime + pd.Timedelta(minutes=1)
    endTime = min([pd.Timestamp(endTime), data.index[-1]])
    tempTime = startTime

    list_used, list_predict = [], []
    index = data.index.tolist().index(tempTime)
    # 每次预测需要当前时间之前的一整天数据（归一化数据）
    inputs = scaler_data[index:index + period]
    # 同时生成历史数据和预测数据
    while tempTime <= endTime:
        # 利用模型进行预测，数据格式为三维
        forecast = model.predict(np.reshape(inputs, (1, period, 1)))
        # 反归一化
        forecast_value = scaler.inverse_transform([[forecast[0, 0]]])[0][0]
        # 得到一个预测数据，上下界按比例得到
        list_predict.append({'predict_time': tempTime, 'timestamp': tempTime.timestamp(),
                             'predict_value': forecast_value, 'max_value': forecast_value * 2.0,
                             'min_value': forecast_value * 0.5})
        # 整理历史数据，并设置报警标志
        list_used.append({'collect_time': tempTime, 'timestamp': tempTime.timestamp(),
                          'used_value': data.loc[tempTime][0],
                          'isAlarm': bool(forecast_value * 0.5 < data.loc[tempTime][0] < forecast_value * 2.0)})
        # 更新模型输入
        inputs = np.concatenate((inputs, forecast), axis=0)[1:]
        # 更新时间
        tempTime = tempTime + pd.Timedelta(minutes=2)

    return {'list_used': list_used, 'list_predict': list_predict}

