import pandas as pd
import random
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    df_test = pd.read_csv('./datasets/1019_100898/y_test.csv')
    df_pred = pd.read_csv('datasets/1019_100898/y_pred_stable.csv')
    y_test = df_test.sort_values(by=['idx'])['altitude'].values
    y_pred = df_pred.sort_values(by=['idx'])['altitude'].values

    mse = mean_squared_error(y_test, y_pred)
    print(mse)
