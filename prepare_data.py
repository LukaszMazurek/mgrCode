
import numpy as np
import pandas as pd
import laspy as lp
import open3d as o3d
import pandas as pd
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.under_sampling import NearMiss, RandomUnderSampler
import os


def get_dataset(point_data, size):
    indices = np.random.choice(point_data.shape[0], size, replace=False)
    return point_data[indices]


def numpy_save_csv(filename, folder,  array):
    filename = f"./datasets/{folder}/{filename}.csv"
    np.savetxt(filename, array, delimiter=',')


def get_full_dataset(train, test):
    return np.concatenate((train, test), axis=0)


if __name__ == '__main__':
    las = lp.read('./full_dataset.las')

    point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))

    num_of_points = len(point_data)

    data = get_dataset(point_data, num_of_points)

    X = np.array([[idx, x[0], x[1]] for idx, x in enumerate(data)])
    y = np.array([[idx, x[2]] for idx, x in enumerate(data)])

    train_size = 0.0075
    train_size_num = int(num_of_points*train_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
    print(len(X_train))
    # X_full = get_full_dataset(X_train, X_test)
    # y_full = get_full_dataset(y_train, y_test)

    X_train = [[x[1], x[2]] for x in X_train]
    y_train = [x[1] for x in y_train]

    test_size = len(X_test)

    dir_name = f'./datasets/{train_size_num}_{test_size}'

    os.mkdir(dir_name)

    pd.DataFrame(X_train, columns=['lat', 'lon']).to_csv(f'{dir_name}/X_train.csv', index=False)
    pd.DataFrame(X_test, columns=['idx', 'lat', 'lon']).to_csv(f'{dir_name}/X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['altitude']).to_csv(f'{dir_name}/y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['idx', 'altitude']).to_csv(f'{dir_name}/y_test.csv', index=False)
