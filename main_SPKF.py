import cv2
import numpy as np
import matplotlib.pyplot as plt
from KalmanFilter.SigmaPoint import SPKF
from scipy.linalg import block_diag

'''
state estimation
x(k+1) = sqrt(5+x(k)) + w(k)
y(k) = x(k)^3 + v(k)
'''
# Define nonlinear function

def f(x,u,w):
    new_x = np.sqrt(5+x) + w
    return new_x

def g(x,u,v):
    y = x**3 + v
    return y

SigmaX = block_diag(1)          # uncertainty of initial state
SigmaW = block_diag(1)          # Process noise covariance
SigmaV = block_diag(2)          # Sensor noise covariance

MY_SPKF = SPKF(f, g, SigmaX, SigmaW, SigmaV)

xtrue = 2 + np.random.rand(1)   # Initialize true system initial state
maxIter = 40
u = 0                           # Unknown initial driving input: assume zero

# % Reserve storage for variables we might want to plot/evaluate
xstore = [xtrue.item()]
xhatstore = []
boundstore = []


for k in range(maxIter):
    # main system
    w = np.random.normal(0, np.sqrt(SigmaW)) 
    v = np.random.normal(0, np.sqrt(SigmaV))
    ytrue = g(xtrue,u,v)
    xtrue = f(xtrue,u,w)

    # estimator
    xhat, xbound = MY_SPKF.iter(u, ytrue)

    xstore.append(xtrue.item())
    xhatstore.append(xhat.item())
    boundstore.append(xbound.item())

xstore = np.array(xstore)
xhatstore = np.array(xhatstore)
boundstore = np.array(boundstore)

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.plot(np.arange(maxIter), xstore[:maxIter], color=(0,0.8,0))
plt.plot(np.arange(maxIter), xhatstore, color=(0,0,1), linestyle='dashed')
plt.fill_between(np.arange(maxIter), xhatstore+boundstore, xhatstore-boundstore, alpha=0.3)
plt.grid()
plt.legend(['true','estimate','bounds'])
plt.title('Sigma-point Kalman filter in action')
plt.xlabel('Iteration')
plt.ylabel('State')

plt.subplot(1,2,2)
estErr = xstore[:maxIter]-xhatstore 
plt.plot(np.arange(maxIter), estErr)
plt.fill_between(np.arange(maxIter), boundstore, -boundstore, alpha=0.3)
plt.grid()
plt.legend(['Error','bounds'])
plt.title('SPKF Error with bounds')
plt.xlabel('Iteration') 
plt.ylabel('Estimation Error')
plt.show()