import numpy as np
import time
import math
nIu = 0.01

def logistic(z):
    return 1 / (1 + np.exp(-z))


def dervtv(z):
    return z * (1 - z)


def read_data(name):
    with open('E:/43/Pattern_off/off_2/Supplied/Supplied/%s' % name) as f:
        # with open('F:/43/Pattern_off/eval/perceptron/trainLinearlyNonSeparable.txt') as f:
        data = []
        labels = []
        for line in f:
            # print(line)
            data.append([float(x) for x in line.split()])

        for dt in data:
            if dt[-1] not in labels:
                labels.append(dt[-1])

    # print(len(data[0]), 'len_data')
    # data = pd.DataFrame(data)
    return data, len(data), len(data[0]) - 1, labels, len(labels)


def process_data(data, len_data, len_attr, clss_c):
    features = []
    classes = []

    for i in range(len_data):
        f = [data[i][j] for j in range(len_attr)]
        f.append(1.00)
        features.append(f)

    features = np.array(features)

    for i in range(len_data):
        l = list(np.zeros(clss_c))
        l[int(data[i][-1])-1] = 1
        classes.append(l)

    classes = np.array(classes)
    return features, classes


tr_d, len_tr_d, len_attributes, classes, class_count = read_data('trainNN.txt')

tr_f, tr_clsses = process_data(tr_d, len_tr_d, len_attributes, class_count)
for i in range(len_attributes):
    tr_f[:, i] = (tr_f[:, i] - np.mean(tr_f[:, i])) / np.std(tr_f[:, i])

tst_d, len_tst_d, len_attributes, classes, class_count = read_data('testNN.txt')
tst_f, tst_clsses = process_data(tst_d, len_tst_d, len_attributes, class_count)
for i in range(len_attributes):
    tst_f[:, i] = (tst_f[:, i] - np.mean(tst_f[:, i])) / np.std(tst_f[:, i])

# print(tst_f)

# print(tr_f)

# hmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm

# tr_f, tr_clsses= process_data(tr_d, len_tr_d, len_attributes, class_count)
# tst_f, tst_clsses= process_data(tst_d, len_tst_d, len_attributes, class_count)


# print(len(tr_f[0]))


# for i in range(len(weights)):
#     print(weights[i].shape)


def forward_propagation(input_data, weights):
    activations = []
    delta = []
    delw = []
    out_put = input_data
    activations.append(out_put)

    for i in range(len(weights)):
        out_put = logistic(np.array(np.dot(out_put, weights[i])))
        delw.append(out_put)
        activations.append(out_put)
        delta.append(out_put)

    labls = activations[-1]
    return delta, delw, activations, labls


def backward_propagation(delta_, delta, delw, activations, weights):
    scaling_f = len(delta_)
    delta[-1] = delta_.copy()

    delw[-1] = np.dot(activations[-2].T, delta[-1])/scaling_f

    for i in range(len(weights) - 1, 0, -1):
        delta[i-1]=np.multiply((np.dot(delta[i], weights[i].T)), dervtv(activations[i]))
        delw[i-1]=(np.dot(activations[i - 1].T, delta[i - 1]))/scaling_f
        # delw[i-1]=-delw[i-1]
    for i in range(len(weights)):
        weights[i]-=nIu*delw[i]


# y, delta, delw, activations=forward_propagation(tr_d)


def train(features, labels, weights):
    start = time.time()
    mine = math.inf

    for i in range(100000):
        delta, delw, activations, y = forward_propagation(features, weights)
        error=(y - labels)*(y - labels)/2
        J= error.sum()
        if J<mine:
            mine=J
        if i%3000==0:
            print(time.time() - start, " ", J)

        delta_ = (y - labels) * dervtv(y)
        backward_propagation(delta_, delta, delw, activations, weights)

    print("Time Taken: ", time.time() - start)

    return weights


def testing(features, labels, weights):
    wrong = 0
    delta, delw, activations, y = forward_propagation(features, weights)
    for i in range(len(y)):
        guess = np.argmax(y[i])
        if labels[i][guess] != 1.00:
            wrong += 1
    return wrong


network_structure = [len(tr_f[0]), 8, 8, len(tr_clsses[0])]
weights = []


for i in range(len(network_structure) - 1):
    w = np.random.rand(network_structure[i], network_structure[i + 1])
    weights.append(np.array(w))


weights=train(tr_f, tr_clsses, weights)
train_missed = testing(tr_f, tr_clsses, weights)
train_accuracy = 100*((len(tr_d)-train_missed)/len(tr_d))
test_missed = testing(tst_f, tst_clsses, weights)
test_accuracy = 100*((len(tst_d)-test_missed)/len(tst_d))

print("Train Data miss-class", train_missed, "Accuracy ", train_accuracy, "%")
print("Test Data miss-class", test_missed, "Accuracy ", test_accuracy, "%")
