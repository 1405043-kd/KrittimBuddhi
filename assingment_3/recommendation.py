import csv
import datetime

import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
from scipy import linalg as LA


def read_data(name):
    data_read = []
    with open(name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data_read.append(row)

    data_read = np.array(data_read, dtype=float)
    data_read = data_read[:, 1:]
    # data_read.dtype = np.float32

    return data_read


def get_error(data, u, v, data_w):
    return np.sum((data_w * (data - np.dot(u, v))) ** 2)


def flip(p):
    return False if random.random() < p else True


def train_valid_test(data):
    train = data.copy()
    test = data.copy()
    valid = data.copy()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if flip(0.6):
                if train[i][j] != 99:
                    train[i][j] = 99
            if flip(0.2):
                if valid[i][j] != 99:
                    valid[i][j] = 99
            if flip(0.2):
                if test[i][j] != 99:
                    test[i][j] = 99
    return train, valid, test


def recommender_trainer(lambda_, k, loop_count, data, data_w, valid, valid_w):
    u = 3 * np.random.rand(len(data), k)
    v = 3 * np.random.rand(k, len(data[0]))
    g_err_diff = 0
    fui = u.copy()
    fvi = v.copy()
    users = len(data)
    items = len(data[0])

    for x in range(loop_count):
        for i in range(users):
            temp_a = np.zeros([k, k])
            temp_b = np.zeros(k)
            # print(temp_b)
            for j in range(items):
                temp_a_t = np.dot(v[:, j].reshape(k, 1), v[:, j].reshape(k, 1).T) + lambda_ * np.eye(k)
                temp_a = temp_a + temp_a_t

                if data[i][j] != 99:
                    temp_b_t = v[:, j].reshape(k, 1) * data[i][j]
                    temp_b = temp_b.reshape(k, 1) + temp_b_t
                    # print(temp_b)
                # print(temp_b)
            u[i, :] = np.linalg.solve(temp_a, temp_b).reshape(1, k)

        # print(np.sum(fui-u))

        v = v.T

        for i in range(items):
            temp_a = np.zeros([k, k])
            temp_b = np.zeros(k)
            # print(temp_b)
            for j in range(users):
                temp_a_t = np.dot(u[j, :].reshape(1, k).T, u[j, :].reshape(1, k)) + lambda_ * np.eye(k)
                temp_a = temp_a + temp_a_t

                if data[j][i] != 99:
                    temp_b_t = u[j, :].reshape(1, k) * data[j][i]
                    temp_b = temp_b.reshape(1, k) + temp_b_t
                    # print(temp_b)
                # print(temp_b)

            v[i, :] = np.linalg.solve(temp_a, temp_b.T).T
            # print(v[:, i].shape)
        v = v.T
        # print(np.sum(fvi - v))

        temp_err = get_error(data, u, v, data_w)
        #####print( g_err_diff-temp_err )

        if abs(temp_err - g_err_diff) < 0.001:
            return get_error(valid, u, v, valid_w), u, v
        g_err_diff = temp_err

            # u[i, :] =  np.dot(v[:,j], test[i][j])).T

        # u = np.linalg.solve(np.dot(v, v.T) + lambda_ * np.eye(n_factors),
        #                     np.dot(v, test.T)).T
        # v = np.linalg.solve(np.dot(u.T, u) + lambda_ * np.eye(n_factors),
        #                     np.dot(u.T, test))
        # g_err = get_error(test, u, v, test_w)

        # print(g_err, g_err-g_err_diff)
        #
        # # if math.floor(g_err-g_err_diff)==0:
        # #     print('chomolokko')
        # #     break
        #
        # g_err_diff = g_err


def recommender_lk_selector(train, train_w, valid, valid_w, test, test_w):
    l_arr = [0.01, 0.1, 1, 10]
    k_arr = [5, 10, 20, 40]
    errors = []
    o_l = 0
    o_k = 0
    min_error = math.inf

    # print(u.shape)
    # print(v.shape)

    for i in range(len(l_arr)):
        for j in range(len(k_arr)):
            errr, u, v = recommender_trainer(l_arr[i], k_arr[j], 100000, train, train_w, valid, valid_w)
            errors.append(errr)
            if errr<min_error:
                min_error=errr
                o_k = k_arr[j]
                o_l = l_arr[i]
                o_u = u
                o_v = v
            with open('file.txt', 'a') as f:
                print(errors, datetime.datetime.now(), file=f)
    with open('file.txt', 'a') as f:
        print(errors, get_error(test, o_u, o_v, test_w), o_k, o_l, file=f)

data = read_data('E:/43/ML_off/ml3/data.csv')
# train, validation = train_test_split(data, test_size=0.2)
# train, test = train_test_split(data, test_size=0.995)
# print(len(train[0]), len(validation[0]), len(test[9][0]))

train, validation, test = train_valid_test(data)

users = len(validation)
items = len(validation[0])
n_factors = 10
lambda_ = 2

print(users, items)

data_w = data.copy()
data_w[data_w == 99] = 0
data_w[data_w != 0] = 1

train_w = train.copy()
train_w[train_w == 99] = 0
train_w[train_w != 0] = 1

validation_w = validation.copy()
validation_w[validation_w == 99] = 0
validation_w[validation_w != 0] = 1

test_w = test.copy()
test_w[test_w == 99] = 0
test_w[test_w != 0] = 1

recommender_lk_selector(train, train_w, validation, validation_w, test, test_w)
# recommender_trainer(lambda_, n_factors, 100, u, v, train, train_w, validation, validation_w)
