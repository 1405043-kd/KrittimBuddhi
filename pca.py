import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mn
from matplotlib import pyplot as MPL
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg as LA


def read_data(name):
    with open('E:/43/ML_off/Assignment2/Assignment2/%s' % name) as f:
        # with open('F:/43/Pattern_off/eval/perceptron/trainLinearlyNonSeparable.txt') as f:
        data = []
        labels = []
        for line in f:
            # print(line)
            data.append([float(x) for x in line.split()])

    # print(len(data[0]), 'len_data')
    # data = pd.DataFrame(data)
    # data_columns = len(data.columns)-1
    #
    # print(data_columns, 'lendata')
    # X = data.ix[:, 0:data_columns].values
    # Y = data.ix[:, data_columns].values
    data = np.array(data)
    # print(len(data))
    # return data, len(data), len(data[0]) - 1, labels, len(labels)
    # return X
    return data


def PCA(data, dimension=2):
    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=False)
    evals, evecs = LA.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evecs = evecs[:, :dimension]
    return np.dot(evecs.T, data.T).T


def plot_pca(data_pca):
    # from matplotlib import pyplot as MPL
    clr1 = '#2026B2'
    # fig = MPL.figure()
    # ax1 = fig.add_subplot(111)
    #
    # # print(data_pca)
    # ax1.plot(data_pca[:, 0], data_pca[:, 1],'.', c=clr1)
    MPL.scatter(data_pca[:, 0], data_pca[:, 1], c=clr1)
    MPL.show()


def Nk(mu, sig, x):
    size = len(x)
    det_sig = np.linalg.det(sig)
    dist_const = 1.0 / (np.math.pow((2 * np.pi), float(size) / 2) * np.math.pow(det_sig, 1.0 / 2))
    xi__mu = np.matrix(x - mu)
    inverse_sig = np.linalg.inv(sig)
    # print(xi__mu)
    dist = dist_const * (np.math.pow(np.math.e, -0.5 * (xi__mu * inverse_sig * xi__mu.T)))
    return dist


def E(data, wks, mu, sig):
    N = len(data)
    K = len(wks)
    P_k = np.zeros([N, K])
    # print(P_k)
    for k in range(K):
        for i in range(N):
            P_k[i][k] = wks[k] * Nk(mu[k], sig[k], data[i, :])
    # print(np.sum(P_k,1))
    return P_k * np.reciprocal(np.sum(P_k, 1)[None].T)


def M(data, wks, mu, sig, P_k):
    N = len(data)
    K = len(wks)

    wks = np.sum(P_k, 0) / N
    # mu1 = data.T.dot(P_k).dot(np.diag(np.reciprocal(np.sum(P_k, 0))))
    # print(mu1.T)
    mu = data.T.dot(P_k) / (np.sum(P_k, 0))
    for k in range(K):
        datMeanSub = data.T - mu[0:, k][None].T.dot(np.ones([1, N]))
        sig[k, :, :] = (datMeanSub.dot(np.diag(P_k[0:, k])).dot(datMeanSub.T)) / np.sum(P_k, 0)[k]
    return wks, mu.T, sig


def log_li(dataSet, wks, Mu, Sigma):
    K = len(wks)
    N, M = np.shape(dataSet)
    # P is an NxK matrix where (i,j)th element represents the likelihood of
    # the ith datapoint to be in jth Cluster (i.e. when z_k = 1)
    P = np.zeros([N, K])
    for k in range(K):
        for i in range(N):
            # print(Mu[k])
            # print(dataSet[i])
            P[i, k] = Nk(Mu[k], Sigma[k], dataSet[i, :])
    return np.sum(np.log(P.dot(wks)))


# print(data)
# print(log_li(data_pca, wk, mu, sig))
# print(Nk(mu[0], sig[0], data_pca[0, :]))

# rv= mn(mu[0], sig[0])
# print(rv.pdf(data_pca[0, :]))

# p_ik=E(data_pca, wk, mu, sig)

# print(M(data_pca, wk, mu, sig, p_ik))
# for itr in range(100):


data = read_data('data.txt')
data_pca = PCA(data)
plot_pca(data_pca)


clusters = 3



mu = np.random.randint(min(data_pca[:, 0]), max(data_pca[:, 0]), size=(clusters, len(
    data_pca[0])))

sig = np.zeros((clusters, len(data_pca[0]), len(data_pca[0])))

for dim in range(len(sig)):
    np.fill_diagonal(sig[dim], np.random.randint(1,3))
wk = np.ones(clusters) / clusters
p_ik = []

log_likelihood_first = 0

for i in range(1000):
    p_ik = E(data_pca, wk, mu, sig)
    wk, mu, sig = M(data_pca, wk, mu, sig, p_ik)
    log_likelihood_second = log_li(data_pca, wk, mu, sig)

    colors = []
    for j in range(len(p_ik)):
        colors.append(np.argmax(p_ik[j]))
    MPL.scatter(data_pca[:, 0], data_pca[:, 1], c=colors)
    MPL.pause(0.1)

    if abs(log_likelihood_second - log_likelihood_first) < 0.0000009:
        print("Converged")
        break
    log_likelihood_first = log_likelihood_second
    print(log_likelihood_second, i)

from collections import Counter
print(Counter(colors))
MPL.show()

# print(data_pca)
