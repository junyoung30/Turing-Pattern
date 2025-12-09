import numpy as np
from tqdm import tqdm

def solver_fft(dt, D, kk, bb):
    coef = 1 / (1 + dt * D * kk)
    bb_hat = np.fft.fft2(bb)
    nv_hat = coef * bb_hat
    return np.fft.ifft2(nv_hat).real

def initialize_grid(Nx, Ny, xleng, yleng, k2, seed):    
    dx, dy = xleng/Nx, yleng/Ny
    
    x = np.linspace(0.5*dx, xleng-0.5*dx, Nx)
    y = np.linspace(0.5*dy, yleng-0.5*dy, Ny)
    xx, yy = np.meshgrid(x, y)
    
    np.random.seed(seed)
    iu = 1 + (k2 ** 2) / 25 + 0.2 * (2 * np.random.rand(*xx.shape) - 1)
    iv = k2 / 5 + 0.2 * (2 * np.random.rand(*xx.shape) - 1)
    return xx, yy, iu, iv, dx, dy

def compute_k_grid(Nx, Ny, dx, dy):
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
    kx, ky = np.meshgrid(kx, ky)
    kk = kx ** 2 + ky ** 2
    return kk

def run_2d_simulation(
    Nx=256, Ny=256, xleng=40, yleng=40,
    T=128, dt=0.035,
    Du=1.0, Dv=0.01, k1=5, k2=11.0, 
    seed=1004, ns=8,
):
    xx, yy, ou, ov, dx, dy = initialize_grid(Nx, Ny, xleng, yleng, k2, seed)
    
    MaxIter = int(T/dt)
    kk = compute_k_grid(Nx, Ny, dx, dy)
    
    tlist, ulist, vlist = [0], [ou.copy()], [ov.copy()]
    
    time = 0
    for it in tqdm(range(MaxIter)):
        time += dt
        
        ff = ou + dt * (k1 * (ov - ou * ov / (1 + ov ** 2)))
        gg = ov + dt * (k2 - ov - 4 * ou * ov / (1 + ov ** 2))
        
        ou = solver_fft(dt, Du, kk, ff)
        ov = solver_fft(dt, Dv, kk, gg)
        
        if (it + 1)%(MaxIter//ns) == 0:
            tlist.append(time)
            ulist.append(ou.copy())
            vlist.append(ov.copy())
    
    parameters = {
        "Du":Du, "Dv":Dv, "k1":k1, "k2":k2, 
        "Nx":Nx, "Ny":Ny, "xleng":xleng, "yleng":yleng,
        "T": T, "dt": dt
    }
    results = {
        "parameters": parameters,
        "tlist": tlist, "ulist": ulist, "vlist": vlist
    }
    
    return results