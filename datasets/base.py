import os
import csv
import numpy as np



def load_boston(return_X_y=True):
    file_path = os.path.dirname(__file__)
    descr_path = os.path.join(file_path, 'descr', 'boston_house_prices.rst')
    with open(descr_path) as f:
        descr_cont = f.read()

    data_path = os.path.join(file_path, 'data', 'boston_house_prices.csv')
    with open(data_path) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        temp = next(data_file)
        feature_names = np.array(temp)
        data = np.zeros((n_samples, n_features))
        target = np.zeros((n_samples,))
        for i,cont in enumerate(data_file):
            data[i] = np.array(cont[:-1], dtype=np.float64)
            target[i] = np.array(cont[-1], dtype=np.float64)

    if return_X_y:
        return data, target

