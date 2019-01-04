import csv
import numpy as np
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


data = read_data('E:/43/ML_off/ml3/data.csv')
train, validation = train_test_split(data, test_size=0.2)
train, test = train_test_split(train, test_size=0.25)
# print(len(train[0]), len(validation[0]), len(test[9][0]))

users = len(validation)
items = len(validation[0])
n_factors = 20


u = 5 * np.random.rand(users, 20)
v = 5 * np.random.rand(20, items)
# print(u.shape)
# print(v.shape)


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

# data_w = np.array(data_w)
# data_w= (data!=99)
# data_w[data_w == False] = 0


print(test)
print(test_w)

# for ii in range(100):
#     print ('lol')


