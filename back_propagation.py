import pandas as pd
import numpy as np


def readData(name):
    with open('E:/43/Pattern_off/off_2/Supplied/Supplied/%s' % name) as f:
        # with open('F:/43/Pattern_off/eval/perceptron/trainLinearlyNonSeparable.txt') as f:
        data = []
        for line in f:
            # print(line)
            data.append([float(x) for x in line.split()])

    # data = pd.DataFrame(data)
    return data


train_data=readData('trainNN.txt')
test_df=readData('testNN.txt')

# print(train_data)


for (x) in train_data:
    print(x[:-1])

