# Nonlinear Kalman Filter Python Implementation

```math
\begin{cases}
x_{k+1}=f(x_k,u_k,w_k)=\sqrt{5+x_k}+w_k \\
y_k=h(x_k,u_k,v_k) =x_k^3+v_k
\end{cases}
```

```math
\Sigma_{\tilde{w}} = 1, \ \ \ \ \Sigma_{\tilde{v}} = 2
```

```bash
pip install -r requirements.txt
```

## Extended Kalman Filter (EKF)

```math
no
```

#### Inference
```bash
python main_EKF.py
```

## Sigma-Point Kalman Filter (SPKF)

#### Inference
```bash
python main_SPKF.py
```

