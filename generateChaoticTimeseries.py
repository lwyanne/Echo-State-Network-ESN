# -*- coding: utf-8 -*-
"""
Generize chaotic time series using auto-updating ESN

Created on Mon Aug  6 14:18:40 2018

@author: Shuying
"""

import numpy as np  
from ESN import *
import matplotlib.pyplot as plt
from function import *
import math



inputs=np.zeros(10000)

autoEsn=ESN(
            n_inputs=1,
            n_outputs=500,
            n_reservoir=500,
            spectral_radius=10,
            ifplot=1,
            alpha=0.8,
            sparsity=0.1
            )

autoEsn.initweights()

autoEsn.update(inputs,1)

#autoEsn.show_internal(shownum=1)
plt.figure()

timeseries=discard(autoEsn.state)


sig=timeseries[44,9700:-2]
sig2=timeseries[44,9702:]
plt.plot(timeseries[44][9800:])

plt.figure()
plt.plot(sig,sig2)
plt.scatter(sig,sig2,marker='.')

plt.title('show trajectory')

plt.show()







#n_reservoir=50 will stable    T=41 why  ????
