import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
from math import *
import image
from mpl_toolkits.mplot3d import Axes3D
from function import *
from ESN import *
import math


# esn=ESN(n_inputs=1,n_outputs=1,ifplot=1,sparsity=0.05)
# esn.initweights()


# def timeshift(dt):
#     u=Lorenz(len=100000)
#     u.downsample(10)
#     u.normalize()
#     u=u.get()
#     u0=u[0][12:-12-dt]
#     utarget=u[0][12+dt:-12]
#     # print('utarget===',utarget)
#     # plt.plot(u0)
#     # plt.plot(utarget)

#     u=Lorenz(init_point=(1,1,3),len=100000)
#     u.downsample(10)
#     u.normalize()
#     u=u.get()
#     utest=u[0][12:-12-dt]
#     utest_target=u[0][12+dt:-12]


#     print('dt=======',dt)
#     print('lenth=============',len(u0))
#     esn.train(u0,utarget,1)
#     esn.predict(utest)
#     esn.err(esn.outputs,utest_target)
#     plt.figure()
#     t=np.arange(0,esn.siglenth)
#     plt.plot(esn.outputs,color='black')
#     plt.plot(utest_target,color='red')
#     return esn.err(esn.outputs,utest_target)
    
# err=[]
# for dt in np.arange(-10,11):
#     err.append(timeshift(dt))
#     print(err)
err=[295.60912302597865,245.18144723228568, 10.888772547097474, 15.412096058138188, 8.1037759534482987, 3.8312916015223508, 0.50188617288938764, 1.2340219229662961, 0.18434224085513634, 0.33480990479688671, 0.21793509382861573, 1.2868466072348357, 4.3514284124729361, 21.457340263785259, 422.39221136781953, 805.34487350417078, 194.21032430960574, 512.90323182697409, 2925.915592466285, 2484.8587280908305, 4251.6436440069592]
err=list(map(math.log,err))
plt.figure()
plt.plot(err)


# print(timeshift(0))
plt.show()



#TODO: 用此刻输入和非此刻输入的问题！！