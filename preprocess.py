import os
import pandas as pd

from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_inputs(data, dump_flag, dump_path, test):
    data_copy = data.copy()
    data_copy['datetime'] = pd.to_datetime(data_copy['datetime'])

    data_copy['month'] = data_copy['datetime'].apply(lambda x: x.month)
    data_copy['day'] = data_copy['datetime'].apply(lambda x: x.day)
    data_copy['hour'] = data_copy['datetime'].apply(lambda x: x.hour)

    data_copy = data_copy.drop('datetime', axis=1)

    # weather_onehot = pd.get_dummies(data_copy['weather'], prefix='weather')
    # data_copy = pd.concat([data_copy, weather_onehot], axis=1)

    # data_copy.drop('weather', axis=1)
    if test:
        return data_copy

    y = data_copy['count']
    X = data_copy.drop('count', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True,
                                                        train_size=0.7,
                                                        random_state=1)

    scaler = StandardScaler()
    scaler.fit(X_train)

    if dump_flag:
        dump(scaler, os.path.join(dump_path, 'scaler.joblib'))

    X_train = pd.DataFrame(data=scaler.transform(X_train), index=X_train.index,
                           columns=X_train.columns)
    X_test = pd.DataFrame(data=scaler.transform(X_test), index=X_test.index,
                          columns=X_test.columns)

    return X_train, X_test, y_train, y_test





