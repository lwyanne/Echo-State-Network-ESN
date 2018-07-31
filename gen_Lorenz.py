"""
Generate a trajectory of the Lorenz system
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from function import *

u=gen_Lorenz(len=10000)
u=downsample(u)
threeDplot(u,'Lorenz')
plt.figure()
plt.plot(u1)
plt.plot(u3)


