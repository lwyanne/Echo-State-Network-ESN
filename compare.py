"""
compare
"""
import numpy as np  
from ESN import *
import matplotlib.pyplot as plt
from function import *
import math

u1=Lorenz((1,1,1),len=100000)  # already discarded the transient
u1.downsample()
u1.normalize()
u1=u1.get()


u2 = Lorenz((5,-3,3),len=100000)
u2.downsample(10)
u2.normalize()
u2=u2.get()



err=[]
para=[]
timeshift=4
N_list=[100,200,400,800]
#x=np.linspace(-8, math.log10(20), num=100)   
x=[0]
for N in N_list:
    u_train=u1[0][12:-12-timeshift]
    u_target=u1[0][12+timeshift:-12]
    u_valid=u2[0][12:-12-timeshift]
    u_true=u2[0][12+timeshift:-12]
    u_true=discard(u_true)
    esn=ESN(n_inputs=1,n_reservoir=N,n_outputs=1,sparsity=0.1,b=1)
    esn.initweights()
    error,lamda=(esn.choose(timeshift,u_train,u_target,u_valid,u_true,x))
    err.append(error)
    para.append(lamda)


err_lesn=[]
for N in N_list:
    u_train=u1[0][12:-12-timeshift]
    u_target=u1[0][12+timeshift:-12]
    u_valid=u2[0][12:-12-timeshift]
    u_true=u2[0][12+timeshift:-12]
    u_true=discard(u_true)
    lesn=LESN(n_inputs=1,n_reservoir=N,n_outputs=1,sparsity=0.1,b=1)
    lesn.initweights()
    lesn.update(u_train,1)
    lesn.allstate=discard(lesn.allstate)
    lesn.fit(u_train,u_target,0,0)
    lesn.update(u_valid,0)
    lesn.allstate=discard(lesn.allstate)
    lesn.predict(1)
    temp=lesn.err(lesn.outputs,u_true,1)
    err_lesn.append(temp)





plt.figure()
plt.plot(N_list,err)
plt.plot(N_list,err_lesn)
print('parameter==================',para)
print('error=====================',err)
plt.title('error with respect to  timeshifts')
plt.show()

