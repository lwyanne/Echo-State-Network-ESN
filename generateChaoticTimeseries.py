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
            n_outputs=50,
            n_reservoir=50,
            spectral_radius=5,
            ifplot=1,
            alpha=0.2,
            sparsity=0.5
            )

autoEsn.initweights()

autoEsn.update(inputs,1)

autoEsn.show_internal()

timeseries=disgard(autoEsn.state)
