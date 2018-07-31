import numpy as np  
from ESN import *
import matplotlib.pyplot as plt
from function import *
import math
import numpy as np

# training samples
u1=Lorenz((1,1,1),len=100000)  # already discarded the transient
u1.downsample()
u1.normalize()
u1=u1.get()

# validation samples
u2 = Lorenz((5,-3,3),len=10000)
u2.downsample(10)
u2.normalize()
u2=u2.get()

# test samples
u3 = Lorenz((2,10,3),len=10000)
u3.downsample(10)
u3.normalize()
u3=u3.get()


esn=ESN(n_inputs=1,n_outputs=1,n_reservoir=1000,sparsity=0.2)
esn.initweights()
x=np.linspace(-8, math.log10(10), num=200) 

u_train=u1[0]
u_target=np.multiply(u_train,u_train)
u_target2=discard(u_target)  

u_valid_in=u2[0]
u_valid_True=np.multiply(u_valid_in,u_valid_in)
u_valid_True2=discard(u_valid_True)

u_test_in=u3[0]
u_test_True=np.multiply(u_test_in,u_test_in)

error_valid=[]
error_train=[]
error_test=[]
para=0

esn.update(u_train,1)
print(np.shape(u_train))
temp1=esn.allstate
temp1=discard(temp1)
print(np.shape(temp1))

esn.update(u_valid_in,0) 
temp2=esn.allstate
temp2=discard(temp2)


for numda in x:
    y=10**numda

    esn.allstate=temp1
    esn.fit(u_train,u_target,y,0)
    
    esn.predict(mode=0)
    #print(np.shape(esn.allstate))
    err=esn.err(esn.outputs,u_target2,0)
    #print(err)
    
    # plt.figure()
    # plt.plot(esn.outputs,color='red')
    # plt.plot(u_target2,color='black')
    # plt.show()
    error_train.append(err)
    del err


    esn.allstate=temp2
    esn.predict(mode=1)
    err=esn.err(esn.outputs,u_valid_True2,0)
    print('numda==',numda,'err==',err)
    error_valid.append(err)

    del err

minE=np.min(error_valid)

para=x[np.argmin(error_valid)]
print(    'choose parameter==',para,
    'minerror_valid===',minE)

plt.figure()
plt.plot(x,error_valid,color='red')
plt.plot(x,error_train,color='black')

esn.mydel()

plt.show()
