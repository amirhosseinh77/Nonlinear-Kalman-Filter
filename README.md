# Nonlinear Kalman Filter Python Implementation

we implement kalman filter for model having following dynamics:

$$\large
\begin{cases}
x_{k+1}=f(x_k,u_k,w_k)=\sqrt{5+x_k}+w_k \\
y_k=h(x_k,u_k,v_k) =x_k^3+v_k
\end{cases}$$


with $\large \Sigma_{\tilde{w}} = 1$ and $\large \Sigma_{\tilde{v}} = 2$ as input and output uncertainty covariance matrices.

```bash
pip install -r requirements.txt
```

## Extended Kalman Filter (EKF)

```math
coming \ soon!
```

#### Inference
```bash
python main_EKF.py
```

## Sigma-Point Kalman Filter (SPKF)

```math
math \ coming \ soon!
```
#### Result
![SPKF_Results](https://github.com/amirhosseinh77/Nonlinear-Kalman-Filter/assets/56114938/1f6b4367-c600-4d67-97fc-d9c40e056815)

#### Inference
```bash
python main_SPKF.py
```

