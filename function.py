"""
all the functions used in this assignment
"""
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from math import *
import image
from mpl_toolkits.mplot3d import Axes3D


def discard(sig):
    sig=np.array(sig)
    l_transient=min(int(np.shape(sig)[-1] / 10), 100)
    if sig.ndim==2:
        temp=sig[:,l_transient:]
    else:
        temp=sig[l_transient:]
    del sig
    return temp

def normal(u):
    """
    This is used to normalize u
    """
    l=len(u)
    temp=[]
    mean=sum(u)/l
    var=sqrt(sum([(x-mean)**2 for x in u])/l)
    for i in u:
        i=(i-mean)/var
        temp.append(i)
    return temp



def solve(A,lamda,ifintercept=0):
    """
    lamda is the regularization parameter. lamda should be non-negetive
    ifintercept=======1 or 0.  1 to change the intercept.  0 to leave the intercept the same.

    using SVD deposition to compute psedoinverse matrix of A
    return the psedoinverse
    """
    if lamda>=0:
        u,s,v=np.linalg.svd(A) 
        regu=lamda*np.eye((np.shape(A.T)[0]))
        #print(regu)
        if not ifintercept:regu[0][0]=0
        singular=np.zeros(np.shape(A.T))
        for i in range(np.shape(s)[0]):
            singular[i][i]=s[i]/(s[i]**2+regu[i][i])
        ans=np.dot(np.dot(v.T,singular),u.T)
    else:
        raise regularizationError('Regularization Parameter Should be Non-negetive !')
      #TODO:np.ones 这里不对！！  
        
    return ans
    
def solve_2(A,lamda,ifintercept):
    if lamda==0: return solve(A,lamda,0)
    else:
        regu=lamda*np.eye(np.shape(A.T)[0])
        #print(regu)
        if not ifintercept:regu[0][0]=0   
        #TODO:
        return np.dot(np.linalg.inv(np.dot(A.T,A)+regu), A.T)



def showMatrix(m):
    """
    convert matrix to image
    """
    fig=plt.figure()
    plt.imshow(m,cmap='gray')
    return

def printMatrix(m):
    """
    to print each element of the matrix
    """
    [rows, cols] = np.shape(m)
    for i in range(rows - 1):
        for j in range(cols-1):
            print(m[j, i])




def threeDplot(signal,label):
    """
    to plot 3D-line-image
    """
    mpl.rcParams['legend.fontsize']=10
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    ax.plot(signal[0],signal[1],signal[2],label='%s'%label)
    ax.legend()
    plt.show()
    return

def plotState(states,shownum,showmode=1):
    """
    showmode=1 : show by row. (plot different rows )
    showmode=0 : show by column (plot different columns)
    
    """
    rd=np.random.randint(0, np.shape(states)[1-showmode], size=shownum)
    rd=list(rd)
    print(rd)
    x=states[rd,:]
    print(np.shape(x))
    timelen=np.shape(x)[-1]
    fig=plt.figure()
    b=np.linspace(0,shownum,shownum)
    a=np.linspace(0,timelen,timelen)
    a,b=np.meshgrid(a,b)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(a,b,x,rstride=1,cstride=0)
    return




def choose(timeshift,esn,x):
    print('**---------------timeshift==%d----------------**'
    %timeshift)
    error=[]
    para=0
    esn.update(u_train,1)
    temp1=esn.allstate
    temp1=discard(temp1)

    esn.update(u_valid,0) 
    temp2=esn.allstate
    print(temp2)
    print(np.shape(temp2))
    temp2=discard(temp2)
    # esn=ESN(n_inputs=1,n_outputs=1,sparsity=0.1)
    # esn.initweights()
    for numda in x:
        y=10**numda
        esn.allstate=temp1
        esn.fit(u_train,u_target,y,0)
        esn.allstate=temp2
        esn.predict(1)
        err=esn.err(esn.outputs,u_true,1)
        print('numda==',numda,'err==',err)
        error.append(err)
    
    minE=np.min(error)

    para=x[np.argmin(error)]
    print('timeshift==',timeshift,
        'choose parameter==',para,
        'minError===',minE)

    plt.figure()
    plt.title('timeshift==%d'%timeshift)
    plt.plot(x,error)
    esn.mydel()
    return minE,para


def choose(object,timeshift,x,fignum):
    #plt.figure()
    #plt.title('timeshift=%d'%(timeshift))
    #plt.plot(u_true,label='true')
    print('**---------------timeshift==%d----------------**'
    %timeshift)
    error=[]
    error_train=[]
    para=0
    object.update(u_train,1)
    temp1=object.allstate
    temp1=discard(temp1)

    object.update(u_valid,0) 
    temp2=object.allstate
    temp2=discard(temp2)
    # esn=ESN(n_inputs=1,n_outputs=1,sparsity=0.1)
    # esn.initweights()
    for lamda in x:
        #plt.figure() 
        #plt.suptitle('timeshift=%d lamda=%f  %s '%(timeshift,lamda,object))
        #plt.subplot(121)
        y=10**lamda
        object.allstate=temp1
        object.fit(u_train,u_target,y,1)
        object.predict(0)
        #plt.plot(object.outputs,label='training outputs with lambda=%f'%lamda)
        #plt.plot(u_target,label='training target')
        #plt.xlabel('time')
        #plt.ylabel('value')

        #plt.legend()
        err=object.err(object.outputs,u_target,1)
        #plt.title('NMSE(log)==%f'%math.log10(err))
        error_train.append(err)
        del err
        del object.outputs
        
        object.allstate=temp2
        object.predict(1)
        err=object.err(object.outputs,u_true,1)

        #plt.subplot(122)
        #plt.plot(u_true,label='true values')
        #plt.plot(object.outputs,label='validation outputs')
        #plt.xlabel('time')
        #plt.ylabel('value')
        #plt.title('NMSE(log)==%f'%math.log10(err))
        #plt.legend()
        print('lamda==',lamda,'err==',err)
        error.append(err)
        
    minE=np.min(error)

    para=x[np.argmin(error)]
    print('timeshift==',timeshift,
        'choose parameter==',para,
        'minError===',minE)


    plt.figure(fignum)
    plt.plot(x,(error_train),label='train_error,timeshift=%d'%timeshift)
    plt.plot(x,(error),label='valid_error,timeshift=%d'%timeshift)
    #esn.mydel()
    return minE,para

    



