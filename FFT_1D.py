import numpy as np


def Solver_fft(dt, D, kk, bb):
    coef = 1 / (1 + dt*D*kk)
    bb_hat = np.fft.fft(bb)
    u_hat = coef * bb_hat
    return np.fft.ifft(u_hat).real


ns = 8
Du, k2 = 1.0, 11.0

Dv, k1 = 0.01, 5

T = 128
dt = 0.5*0.07
MaxIter = round(T/dt)

Nx = 256
xleng = 40
dx = xleng/Nx

x = np.linspace(0.5*dx, xleng-0.5*dx, Nx)
    
np.random.seed(1)
iu = 1+(k2**2)/25+0.2*(2*np.random.rand(Nx)-1)
iv = k2/5+0.2*(2*np.random.rand(Nx)-1)

k = 2*np.pi * np.fft.fftfreq(Nx, dx)
kk = k**2

ou, ov = iu.copy(), iv.copy()

tlist, ulist, vlist = [], [ou.copy()], [ov.copy()]

time = 0
tlist.append(time)

for it in range(MaxIter):
    time += dt
    
    ff = ou+dt*k1*(ov-ou*ov/(1+ov**2))
    gg = ov+dt*(k2-ov-4*ou*ov/(1+ov**2))
    
    ou = Solver_fft(dt, Du, kk, ff)
    ov = Solver_fft(dt, Dv, kk, gg)
    
    if (it+1)%(MaxIter//ns) == 0:
        tlist.append((it+1)*dt)
        ulist.append(ou.copy())
        vlist.append(ov.copy())