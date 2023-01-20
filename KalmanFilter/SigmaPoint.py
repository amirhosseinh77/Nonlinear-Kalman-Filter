import numpy as np 
from scipy.linalg import block_diag, cholesky

class SPKF:
    def __init__(self, f, g, SigmaX0, SigmaW, SigmaV):
        # store model
        self.f = f
        self.g = g
        
        # initial state
        self.xhat  = np.array(0)

        # Covariance values
        self.SigmaX = SigmaX0
        self.SigmaW = SigmaW
        self.SigmaV = SigmaV

        # SPKF specific parameters
        self.Nx = self.SigmaX.shape[0]
        self.Nw = self.SigmaW.shape[0]
        self.Nv = self.SigmaV.shape[0]
        self.Na = self.Nx+self.Nw+self.Nv

        # CDKF bar weights
        Wmx = np.zeros(2*self.Na+1)
        self.h = np.sqrt(3)
        Wmx[0] = (self.h**2-self.Na)/(self.h**2)
        Wmx[1:] = 1/(2*self.h**2) 
        Wcx=Wmx
        self.Wm = Wmx.reshape(-1,1) # mean
        self.Wc = Wcx.reshape(-1,1) # covar

    def iter(self, input, output):
        # Step 1a-1: Create augme5nted xhat and SigmaX
        sigmaXa = block_diag(self.SigmaX, self.SigmaW, self.SigmaV)
        sigmaXa = np.real(cholesky(sigmaXa, lower=True))
        xhata = np.vstack([self.xhat, np.zeros((self.Nw+self.Nv,1))])
        
        # Step 1a-2: Calculate SigmaX points
        Xa = xhata.reshape(self.Na,1) + self.h*np.hstack([np.zeros((self.Na, 1)), sigmaXa, -sigmaXa])

        # Step 1a-3: Time update from last iteration until now
        Xx = self.f(Xa[:self.Nx,:], input, Xa[self.Nx:self.Nx+self.Nw,:])
        Xx = np.vstack(Xx)
        xhat = Xx@self.Wm

        # Step 1b: Error covariance time update
        #          - Compute weighted covariance sigmaminus(k)
        SigmaX = (Xx - xhat)@np.diag(self.Wc.ravel())@(Xx - xhat).T

        # Step 1c: Output estimate
        #          - Compute weighted output estimate yhat(k)
        Y = self.g(Xx, input, Xa[self.Nx+self.Nw:,:])
        yhat = Y@self.Wm
        
        # Step 2a: Estimator gain matrix
        SigmaXY = (Xx - xhat)@np.diag(self.Wc.ravel())@(Y - yhat).T
        SigmaY  = (Y - yhat)@np.diag(self.Wc.ravel())@(Y - yhat).T
        L = SigmaXY/SigmaY
        
        # Step 2b: State estimate measurement update
        xhat = xhat + L*(output - yhat)

        # Step 2c: Error covariance measurement update
        SigmaX = SigmaX - L@SigmaY@L.T
        _,S,V = np.linalg.svd(SigmaX)
        HH = V.T@np.diag(S)@V
        SigmaX = (SigmaX + SigmaX.T + HH + HH.T)/4 # Help maintain robustness
        
        # Save data for next iteration...
        self.SigmaX = SigmaX
        self.xhat = xhat
        
        xk = self.xhat[0]
        xkbnd = 3*np.sqrt(SigmaX[0, 0])
        return xk, xkbnd