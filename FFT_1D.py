import numpy as np

def Solver_fft(dt, D, kk, bb):
    coef = 1 / (1 + dt*D*kk)
    bb_hat = np.fft.fft(bb)
    u_hat = coef * bb_hat
    return np.fft.ifft(u_hat).real