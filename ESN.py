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




class Lorenz():
    """
    This is used to generate trajectory of Lorenz System.
    (Interate by Heun Method)
    h===========the integration step size
    len=========the length of time series 

    """
    
    def __init__(self,init_point=(1,1,1),len=200000,r=28,sigma=10,beta=8/3.0,h=0.005):
        self.length=len
        self.r=r
        self.sigma=sigma
        self.beta=beta
        self.h=h    
        self.u=np.zeros((3,len))
        self.u[:,0]=list(init_point)
        #using Heun method for numerical solution
        for i in range(1,len):
            temp0=self.u[0][i-1]+self.h*self.sigma*(self.u[1][i-1]-self.u[0][i-1])    
            temp1=self.u[1][i-1]+self.h*(self.u[0][i-1]*(self.r-self.u[2][i-1])-self.u[1][i-1])
            temp2=self.u[2][i-1]+self.h*(self.u[0][i-1]*self.u[1][i-1]-self.beta*self.u[2][i-1])
            self.u[0][i]=self.u[0][i-1]+self.h/2*(self.sigma*(self.u[1][i-1]-self.u[0][i-1])+self.sigma*(temp1-temp0))
            self.u[1][i]=self.u[1][i-1]+self.h/2*(self.u[0][i-1]*(self.r-self.u[2][i-1])-self.u[1][i-1]+temp0*(self.r-temp2)-temp1)
            self.u[2][i]=self.u[2][i-1]+self.h/2*(self.u[0][i-1]*self.u[1][i-1]-self.beta*self.u[2][i-1]+temp0*temp1-self.beta*temp2)

        #discard the Transient 10s
        for i in range(3):
            temp=self.u
            del self.u
            self.u=temp[:,2000:]
    
    def downsample(self,interval=10):
        self.interval=interval
        r=[]
        for j in range(np.shape(self.u)[0]):
            r.append([])
            for i in range(len(self.u[j])):
                if i% self.interval ==0:
                    r[j].append(self.u[j][i]) 
        self.u=r

    def normalize(self):
        for i in range(3):
            self.u[i]=normal(self.u[i])


    def show(self):
        mpl.rcParams['legend.fontsize']=10
        fig=plt.figure()
        ax=fig.gca(projection='3d')
        ax.plot(self.u[0],self.u[1],self.u[2])
        ax.legend()
        plt.title('Lorenz time series , each step is %f'%(self.h*self.interval))
        plt.show()
    
    def get(self):
        return self.u

        

class ESN():

    def __init__(self,n_inputs,n_outputs,
                 n_reservoir=200,
                spectral_radius=0.95,
                sparsity=0.1, 
                ifplot=0,
                noise=0.001,
                seednum=42,
                b=0.35,
                a=1,
                alpha=0):
        """
        Args:
            n_inputs: the number of input dimensions
            n_outputs: the number of output dimensions
            n_reservoir:nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
        """

        self.n_inputs=n_inputs
        self.n_reservoir=n_reservoir
        self.n_outputs=n_outputs
        self.spectral_radius=spectral_radius
        self.sparsity=sparsity
        self.noise=noise
        self.flag=self.n_inputs-1
        self.seednum=seednum
        self.b=b
        self.a=a
        self.ifplot=ifplot
        self.alpha=alpha
    
    def initweights(self):
        """
        initialize recurrent weights:
        """
        np.random.seed(self.seednum)

        W=np.random.normal(size=(self.n_reservoir,self.n_reservoir))

        n_zero = round(self.n_reservoir*self.sparsity) # the number of zero elements in each

        for i in range(self.n_reservoir):
            self.dic=list(map(
                round,(self.n_reservoir*np.random.rand(n_zero))))
            for j in range(self.n_reservoir):
                if j not in self.dic:
                    W[i,j]=0
            del self.dic

        radius=np.max(np.abs(scipy.linalg.eigvals(W)))
        self.W=W*self.spectral_radius/radius


        #initialize input weights:
        np.random.seed(self.seednum+1)
        V=np.random.normal(size=(self.n_reservoir,self.n_inputs))
        np.random.seed(self.seednum+2)        
        dic=list(map(round,(self.n_reservoir*np.random.rand(n_zero))))

        for i in range(self.n_inputs):
            for j in range(self.n_reservoir):
                if j not in dic:
                    V[j,i]=0
                    
        del dic
        self.V=V
        #initialize bias weights:
        np.random.seed(self.seednum+4)        
        self.Wb=self.b*np.random.normal(size=(self.n_reservoir,1))
        np.random.seed(self.seednum+5)
        self.noiseVec=self.noise * np.random.normal(size=(self.n_reservoir,1)).T    
        
    def update(self,inputs,ifrestart):
        """
        update the state of internal nodes.
        """
        self.lenth=int(np.size(inputs)/self.n_inputs)
        inputs=np.reshape(inputs, (self.n_inputs,self.lenth))  

        if ifrestart:
            self.state=np.zeros((self.n_reservoir,self.lenth))
            self.state[:,0]=np.dot(self.V,inputs[:,0].T)

        else:
            self.state=np.hstack(
                (self.laststate,
                np.zeros((self.n_reservoir,self.lenth)))
                )      

            inputs=np.hstack((self.lastinput,inputs))

            self.lenth+=1       

        for i in range(1, self.lenth):
            self.state[:,i]=(
                self.alpha * self.state[:,i]
                +
                (1 - self.alpha) * np.tanh(
                    np.dot(self.W.T, self.state[:,i-1])
                        + self.a * np.dot(self.V, inputs[:,i].T) 
                        + self.Wb.T
                    ) 
                    + self.noiseVec  #TODO:
                    #+ self.noise * np.random.normal(size=(self.n_reservoir,1)).T   
            )       

        self.laststate=self.state[:,-1]
        self.laststate=np.reshape(self.laststate, (len(self.laststate),-1))
        self.lastinput=inputs[:,-1]
        self.lastinput=np.reshape(self.lastinput, (len(self.lastinput),-1))
        self.bias=np.ones((1,self.lenth))
        self.allstate=np.vstack((self.bias,self.state))

    def show_internal(self, shownum=5):
        rd=np. random. randint(0, self. n_reservoir - 1, size=(shownum, 1))
        
        
        
        

    def fit(self, inputs,targets,namda,ifintercept=0):
        """
        fit the output weights, using Ridge Regression methods.
        return: the coefficience matrix
        NOTICE: namda should be non-negative
        """   
        targets=np.array(targets) 
        inputs=np.array(inputs)       
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (-1,len(inputs)))
        if targets.ndim < 2:
            targets = np.reshape(targets, (-1,len(targets)))
        # self.update(inputs,ifrestart) 
        #self.state=discard(self.state)
        #targets=discard(targets)
        #self.bias=np.ones((1,np.shape(targets)[-1]))
        #self.allstate=np.vstack((self.bias,self.state))
        self.coefs=np.dot(solve_2(self.allstate.T,namda,ifintercept),targets.T)
    
 
  
     
    def choose(self,timeshift,u_train,u_target,u_valid,u_true,x):
        print('**---------------timeshift==%d----------------**'
        %timeshift)
        error=[]
        para=0
        self.update(u_train,1)
        temp1=self.allstate
        temp1=discard(temp1)

        self.update(u_valid,0) 
        temp2=self.allstate
        print(temp2)
        print(np.shape(temp2))
        temp2=discard(temp2)
        # esn=ESN(n_inputs=1,n_outputs=1,sparsity=0.1)
        # esn.initweights()
        for numda in x:
            y=10**numda
            self.allstate=temp1
            self.fit(u_train,u_target,y,0)
            self.allstate=temp2
            self.predict(1)
            err=self.err(self.outputs,u_true,1)
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
        self.mydel()
        return minE,para

    def crossValid(self,numda):
        
        pass #TODO:


    def predict(self,mode):
        """
        mode=1 or 0
        """
        #print('inputs',np.shape(inputs))
        #self.update(inputs,0)
        self.outputs=np.dot(self.allstate.T,self.coefs)[mode:]
        #print('outputs',np.shape(self.outputs))


    

    def err(self,signal,real,ifnormal):
        """
        calculate the error.
        ifnormal===1 :  use Normalized Mean Square Error
        ifnormal===0:   use Mean Square Error
        """
        # get the length
        self.siglenth=int(np.size(signal)/self.n_inputs)
        # reshape the two signals in case one is column vector, 
        # and the other is row vector
        real=np.reshape(np.array(real),(self.siglenth,1))
        signal=np.reshape(np.array(signal),(self.siglenth,1))     
        if ifnormal:
            err=(np.mean(
                    np.multiply(
                            (signal-real),(signal-real)
                            ))/
                    np.mean(np.multiply(
                            real-np.mean(real),real-np.mean(real)
                            )
            ))
        
        else:
            err=(np.sum(np.multiply((signal-real),(signal-real))
        ))/self.siglenth

        return err

    def test(self,inputs,targets,ratio):
        print('targets===',targets)
        train_len=int((np.size(inputs)/self.n_inputs)*ratio)
        print('train_len==',train_len)
        train_samples=inputs[:train_len-1]
        train_targets=targets[:train_len-1]
        self.test_samples=inputs[train_len:]
        self.test_real=targets[train_len:]
        print(self.test_real)
        self.train(train_samples,train_targets,1)   
        self.predict(self.test_samples)
        plotState(self.state,5)
        self.erro=self.err(self.outputs,self.test_real)
        return self.erro


    def mydel(self):
        del self.allstate
        del self.state
        del self.bias
        del self.coefs
        del self.outputs
        
    


class LESN(ESN):
    def update(self,inputs,ifrestart):
        """
        update the state of internal nodes.
        """
        self.lenth=int(np.size(inputs)/self.n_inputs)
        inputs=np.reshape(inputs, (self.n_inputs,self.lenth))  

        if ifrestart:
            self.state=np.zeros((self.n_reservoir,self.lenth))
            self.state[:,0]=np.dot(self.V,inputs[:,0].T)

        else:
            self.state=np.hstack(
                (self.laststate,
                np.zeros((self.n_reservoir,self.lenth)))
                )      

            inputs=np.hstack((self.lastinput,inputs))

            self.lenth+=1       

        for i in range(1,self.lenth):
            self.state[:,i]=(
                    np.dot(self.W.T, self.state[:,i-1])
                        + self.a * np.dot(self.V, inputs[:,i].T) 
                        + self.Wb.T
                    + self.noiseVec  #TODO:
                    #+ self.noise * np.random.normal(size=(self.n_reservoir,1)).T   
            )       

        self.laststate=self.state[:,-1]
        self.laststate=np.reshape(self.laststate, (len(self.laststate),-1))
        
        self.lastinput=inputs[:,-1]
        self.lastinput=np.reshape(self.lastinput, (len(self.lastinput),-1))
        self.bias=np.ones((1,self.lenth))
        self.allstate=np.vstack((self.bias,self.state))


class trivial(ESN):
    pass

def test1():
    u=Lorenz(len=10000)
    u.downsample(10)
    u.normalize()
    u=u.get()
    plt.figure()
    plt.plot(u[0])
    plt.figure()
    plt.plot(u[2])
    esn=ESN(n_inputs=1,n_outputs=1,n_reservoir=100,ifplot=1)
    esn.initweights()
    print(np.shape(u))    
    erro=esn.test(u[0],u[2],0.8)
    print('error====',erro)
    plt.figure()
    t=np.arange(0,esn.siglenth)
    plt.plot(esn.outputs,color='black')
    plt.plot(esn.test_real,color='red')
    def get(self):
        return self.u

    plt.show()


def test_timeShift(shift):
    u=gen_Lorenz(len=1000) 
    u=downsample(u)
    u=np.array(u)    
    plt.plot(u[0])
    u0=normal(u[0,20:])
    U=u0[shift:200+shift]
    u0=u0[0:200]
    utarget=np.array(U) 
    esn=ESN(n_inputs=1,n_outputs=1,n_reservoir=200,ifplot=1)
    esn.initweights()
    esn.test(u0,utarget,0.7)
    plt.figure()
    t=np.arange(0,esn.siglenth)
    plt.plot(esn.outputs,color='black')
    plt.plot(esn.test_real,color='red')
    plt.show()





