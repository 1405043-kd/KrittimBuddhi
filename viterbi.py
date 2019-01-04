import numpy as np

def readfile(filename):
    file = open(filename, 'rb')

    initial = [0, 0, int(file.read(1))]

    tuples = [initial.copy()]
    while True:
        next_bit = file.read(1)
        initial.pop(0)
        if not next_bit:
            break

        initial.append(int(next_bit))
        tuples.append(initial.copy())
    return tuples


def read_data(name):
    with open(name) as f:
        # with open('F:/43/Pattern_off/eval/perceptron/trainLinearlyNonSeparable.txt') as f:
        data = []
        labels = []
        for line in f:
            # print(line)
            data.append([float(x) for x in line.split()])

    return data


# tuples = readfile('E:/43/ML_off/Assignment2/Assignment2/train.txt')


h = read_data('E:/43/ML_off/Assignment2/Assignment2/coef.txt')
# print(h[0][0])
h1= h[0][0]
h2= h[1][0]
# print(h1, h2)

x_ks=[[],[],[],[],[],[],[],[]]

def tuple_no(tuple):
    return (int((str(tuple[0])+str(tuple[1])+str(tuple[2])),2))

# for i in range(len(tuples)):
#     temp= tuples[i]
#     t_tuple_no=tuple_no(temp)
#     nk = np.random.normal(.5, .3, 1)[0]
#     # print(nk)
#     x_k=h1*temp[2] + h2*temp[1]+nk
#     x_k_1 = h1 * temp[1] + h2 * temp[0]+nk
#     x_ks[t_tuple_no].append(([x_k, x_k_1]))
# print(x_ks)

file = open('E:/43/ML_off/Assignment2/Assignment2/train.txt', 'rb')

initial = [0, 0, int(file.read(1))]

tuples = [initial.copy()]
while True:
    next_bit = file.read(1)
    tuples.pop(0)
    if not next_bit:
        break

    initial.append(int(next_bit))

    t_tuple_no=tuple_no(initial)
    nk = np.random.normal(.5, .3, 1)[0]
    # print(nk)
    x_k=h1*tuples[2] + h2*tuples[1]+nk
    x_k_1 = h1 * tuples[1] + h2 * tuples[0]+nk
    x_ks[t_tuple_no].append(([x_k, x_k_1]))
