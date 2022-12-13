from sklearn.preprocessing import MinMaxScaler

Scaler = MinMaxScaler(feature_range=(0, 1))


def normalize_data(data):
    for i in range(len(data)):
        data[i][0] = Scaler.fit_transform(data[i][0])
    return data
