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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            # print(b, w)

            a = sigmoid(np.dot(w, a) + b)
            # print(a)
            # print('\n')
        return a

    # def backprop(self, x, y):
    #     """Return a tuple ``(nabla_b, nabla_w)`` representing the
    #     gradient for the cost function C_x.  ``nabla_b`` and
    #     ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    #     to ``self.biases`` and ``self.weights``."""
    #     nabla_b = [np.zeros(b.shape) for b in self.biases]
    #     nabla_w = [np.zeros(w.shape) for w in self.weights]
    #     # feedforward
    #     activation = x
    #     activations = [x]  # list to store all the activations, layer by layer
    #     zs = []  # list to store all the z vectors, layer by layer
    #     for b, w in zip(self.biases, self.weights):
    #         z = np.dot(w, activation) + b
    #         zs.append(z)
    #         activation = sigmoid(z)
    #         activations.append(activation)
    #     # backward pass
    #     delta = self.cost_derivative(activations[-1], y) * \
    #             sigmoid_prime(zs[-1])
    #     nabla_b[-1] = delta
    #     nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    #     # Note that the variable l in the loop below is used a little
    #     # differently to the notation in Chapter 2 of the book.  Here,
    #     # l = 1 means the last layer of neurons, l = 2 is the
    #     # second-last layer, and so on.  It's a renumbering of the
    #     # scheme in the book, used here to take advantage of the fact
    #     # that Python can use negative indices in lists.
    #     for l in xrange(2, self.num_layers):
    #         z = zs[-l]
    #         sp = sigmoid_prime(z)
    #         delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
    #         nabla_b[-l] = delta
    #         nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
    #     return (nabla_b, nabla_w)

    # def evaluate(self, test_data):
    #     """Return the number of test inputs for which the neural
    #     network outputs the correct result. Note that the neural
    #     network's output is assumed to be the index of whichever
    #     neuron in the final layer has the highest activation."""
    #     test_results = [(np.argmax(self.feedforward(x)), y)
    #                     for (x, y) in test_data]
    #     return sum(int(x == y) for (x, y) in test_results)

    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x[:-1])), x[-1])
                        for x in test_data]
        for (x,y) in test_results:
            print(x,y)
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)

a=4
net = Network([4, 8, 8, a])

# print(net.biases)
# print((net.feedforward(train_data[0][:-1])))
# print(np.argmax(net.feedforward(train_data[1][:-1])))
print(net.evaluate(train_data))
