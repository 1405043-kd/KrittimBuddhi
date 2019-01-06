# from sklearn.datasets.samples_generator import make_blobs
# from matplotlib import pyplot as MPL
# import numpy as np
#
# X,Y = make_blobs(cluster_std=1, random_state=40,n_samples=500,centers=3)
# X = np.dot(X,np.random.RandomState(0).randn(2,2))
#
# mu = np.random.randint(min(X[:, 0]), max(X[:, 0]), size=(3, len(
#     X[0])))
# print(mu)
# print(X[0])
# print(X[1,:])
#
# # # print(Y)
# # fig = MPL.figure()
# # ax1 = fig.add_subplot(111)
# # clr1 = '#2026B2'
# # ax1.plot(X[:, 0], X[:, 1], '.', mfc=clr1, mec=clr1)
# # MPL.show()
# #
#
#
# # mu = np.random.randint(min(X[:, 0]), max(X[:, 0]), size=(3, len(
# #             X[0])))
# # cov = np.zeros((3,len(X[0]),len(X[0])))
# # for dim in range(len(cov)):
# #             np.fill_diagonal(cov[dim],5)
# # pi = np.ones(3) / 3
# # print(pi)
#




# import numpy as np
# w=[[ 2.23168657,  0.11165161, -0.60512351, -0.57749564,],
#  [ 0.99412542,  2.07679486,  0.01753047, -0.02170952],
#  [ 0.35944073, -0.02779284,  0.52560112,  0.22054527],
#  [ 0.1877743  , 0.13534587,  0.69647307 , 0.57577417],
#  [ 2.13094846 ,-0.52338678 , 0.49555865 ,-0.71486488],
#  [-0.46549805 ,-0.20261216  ,1.04164241 ,-0.18684562],
#  [-0.08527528 ,-0.79357736, -0.59472859  ,1.14115563],
#  [-1.37215729 ,-0.20383003 , 0.6299625  , 1.48082712]]
#
# a=[1,2,3,4]
#
# for (x,y,z,f) in a:
#     print(x,y,z,f)
#
# b=[[ 0.76715421],
#  [ 0.39660832],
#  [ 1.13662187],
#  [ 0.24457126],
#  [-0.18580615],
#  [ 0.32522664],
#  [-0.60497479],
#  [ 2.1565148 ],]
#
# for i in range(len(w)-2, -1, -1):
#     print(i)
#
# # print(np.dot(w,a))
# #
# # print('\n')
# #
# # print(np.dot(w,a)+b)

# print(np.ones(3)/4)

#
# initial = [0, 0, 1]
#
# windows = [initial]
# print(initial)

import numpy as np
# import pickle
# u =  np.random.rand(10, 3)
# v =  np.random.rand(3, 10)
#
# with open('test.pkl', 'wb') as f:
#     pickle.dump(np.dot(u,v), f)
#
# print(np.dot(u, v))
#
# with open('test.pkl','rb') as f:
#     x = pickle.load(f)
#     # print(x.shape)
#     print(u)
i_k = np.eye(10)
print(i_k)