import numpy as np
import os

from joblib import dump
from sklearn.ensemble import RandomForestRegressor


def train(X_train, y_train):
    model = RandomForestRegressor(n_estimators=300, bootstrap=True)
    model.fit(X_train, y_train)
    return model


def test(X_test, y_test, model, dump_flag, dump_path):
    #predict on test data
    y_pred = model.predict(X_test)

    #evaluate performance
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum(
        (y_test - y_test.mean())**2))

    print("RMSE: {:.2f}".format(rmse))
    print(" R^2: {:.4f}".format(r2))

    if dump_flag:
        dump(model, os.path.join(dump_path, 'model.joblib'))
