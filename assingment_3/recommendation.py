import csv
import datetime
import pickle

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


def get_error(data, u, v):
    # return np.sum((data_w * (data - np.dot(u, v))) ** 2)
    sum = 0
    counte = 0
    uv = np.dot(u, v)
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] != 99:
                sum += ((data[i][j] - uv[i][j]) ** 2)
                counte = counte + 1
    return math.sqrt(sum / counte)


def get_error_uv(data, uv):
    # return np.sum((data_w * (data - np.dot(u, v))) ** 2)
    sum = 0
    counte = 0
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] != 99:
                sum += ((data[i][j] - uv[i][j]) ** 2)
                counte = counte + 1
    return math.sqrt(sum / counte)


def flip(p):
    return False if np.random.uniform(0, 1) < p else True


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


def recommender_trainer(lambda_, k, loop_count, data):
    np.random.seed(103)
    u = 2 * np.random.rand(len(data), k)
    v = 3 * np.random.rand(k, len(data[0]))
    i_k = np.eye(k)
    err_prev = 0
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
                temp_a_t = np.dot(v[:, j].reshape(k, 1), v[:, j].reshape(k, 1).T) + lambda_ * i_k
                temp_a = temp_a + temp_a_t

                if data[i][j] != 99:
                    temp_b_t = v[:, j].reshape(k, 1) * data[i][j]
                    # temp_b_t = np.multiply(v[:, j].reshape(k, 1), data[i][j])
                    temp_b = temp_b.reshape(k, 1) + temp_b_t
                    # print(temp_b)
                # print(temp_b)
            # u[i, :] = np.linalg.solve(temp_a, temp_b)
            u[i, :] = np.dot(np.linalg.inv(temp_a), temp_b).reshape(1, k)

        # print(np.sum(fui-u))

        v = v.T

        for i in range(items):
            temp_a = np.zeros([k, k])
            temp_b = np.zeros(k)
            # print(temp_b)
            for j in range(users):
                temp_a_t = np.dot(u[j, :].reshape(1, k).T, u[j, :].reshape(1, k)) + lambda_ * i_k
                temp_a = temp_a + temp_a_t

                if data[j][i] != 99:
                    temp_b_t = u[j, :].reshape(1, k) * data[j][i]
                    # temp_b_t = np.multiply(u[j, :].reshape(1, k),data[j][i])
                    temp_b = temp_b.reshape(1, k) + temp_b_t
                    # print(temp_b)
                # print(temp_b)

            # v[i, :] = np.linalg.solve(temp_a, temp_b.T).T
            v[i, :] = np.dot(np.linalg.inv(temp_a), temp_b.T).T
            # print(v[:, i].shape)
        v = v.T
        # print(np.sum(fvi - v), np.sum(fui - u), "u, v")

        err_curr = get_error(train, u, v)
        print(abs(err_prev - err_curr))

        if abs((err_prev - err_curr) / err_curr) < 0.001:
            return u, v
        err_prev = err_curr

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


def marger(data1, data2):
    for i in range(len(data1)):
        for j in range(len(data1[0])):
            if data1[i][j] == 99 and data2[i][j] != 99:
                data1[i][j] = data2[i][j]
    return data1


def recommender_lk_selector(train, valid, test):
    l_arr = [0.1, 0.01, 10, 1]
    k_arr = [20, 40, 5, 10]
    errors = []
    o_l = 0
    o_k = 0
    min_error = math.inf

    # print(u.shape)
    # print(v.shape)
    best = 0
    for i in range(len(l_arr)):
        for j in range(len(k_arr)):
            u, v = recommender_trainer(l_arr[i], k_arr[j], 100000, train)
            errr = get_error(valid, u, v)
            errors.append(errr)
            if errr < min_error:
                # print("yaia ", k_arr[j], l_arr[i])
                min_error = errr
                o_k = k_arr[j]
                o_l = l_arr[i]
                # best = l_arr[i] + k_arr[j]
            with open('file.txt', 'a') as f:
                print(errors, datetime.datetime.now(), file=f)

    marged_data = marger(train, valid)
    o_u, o_v = recommender_trainer(o_l, o_k, 100000, marged_data)

    with open('file.txt', 'a') as f:
        print(errors,  o_k, o_l,"Error Train: " ,get_error(marged_data, o_u, o_v),"Error Test: ", get_error(test, o_u, o_v), file=f)

    with open('test.pkl', 'wb') as ff:
        pickle.dump(np.dot(o_u, o_v), ff)

    # print(np.dot(o_u, o_v))

    # with open('test.pkl', 'rb') as f:
    #     x = pickle.load(f)
    return o_l, o_k


def rec_eng(test):
    with open('test_best.pkl', 'rb') as f:
        x = pickle.load(f)
    print("test: ", get_error_uv(test, x))


def test_with_best(train, validation, l, k):
    marged_data = marger(train, validation)
    o_u, o_v = recommender_trainer(l, k, 100000, marged_data)
    print(get_error(marged_data, o_u, o_v))

    with open('test_best.pkl', 'wb') as ff:
        pickle.dump(np.dot(o_u, o_v), ff)


data = read_data('E:/43/ML_off/ml3/data.csv')
# train, validation = train_test_split(data, test_size=0.2)
# train, test = train_test_split(data, test_size=0.998)
# print(len(train[0]), len(validation[0]), len(test[9][0]))

train, validation, test = train_valid_test(data)

users = len(validation)
items = len(validation[0])

print(np.sum(train - validation), np.sum(train - test), np.sum(validation - test))
print(users, items)

#
# data_w = data.copy()
# data_w[data_w == 99] = 0
# data_w[data_w != 0] = 1
#
# train_w = train.copy()
# train_w[train_w == 99] = 0
# train_w[train_w != 0] = 1
#
# validation_w = validation.copy()
# validation_w[validation_w == 99] = 0
# validation_w[validation_w != 0] = 1
#
# test_w = test.copy()
# test_w[test_w == 99] = 0
# test_w[test_w != 0] = 1

l, k = recommender_lk_selector(train, validation, test)
# recommender_trainer(lambda_, n_factors, 100, u, v, train, train_w, validation, validation_w)

test_with_best(train, validation, l, k)
rec_eng(test)