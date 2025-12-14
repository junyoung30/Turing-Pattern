import numpy as np
from tqdm import tqdm
import time

from .base import PatternSolver



class FFTSolverNumpy(PatternSolver):
    
    @staticmethod
    def initialize_grid(Nx, Ny, xleng, yleng, k2, seed):    
        dx, dy = xleng/Nx, yleng/Ny

        x = np.linspace(0.5*dx, xleng-0.5*dx, Nx)
        y = np.linspace(0.5*dy, yleng-0.5*dy, Ny)
        xx, yy = np.meshgrid(x, y)

        np.random.seed(seed)
        iu = 1.0 + (k2**2) / 25.0 + 0.2*(2*np.random.rand(*xx.shape)-1)
        iv = k2/5.0 + 0.2*(2*np.random.rand(*xx.shape)-1)
        return iu, iv, dx, dy
        
    @staticmethod
    def solver_fft(dt, D, kk, bb):
        coef = 1 / (1 + dt * D * kk)
        bb_hat = np.fft.fft2(bb)
        nv_hat = coef * bb_hat
        return np.fft.ifft2(nv_hat).real
    
    @staticmethod
    def compute_k_grid(Nx, Ny, dx, dy):
        kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
        kx, ky = np.meshgrid(kx, ky)
        kk = kx ** 2 + ky ** 2
        return kk

    
    def solve(self, params, seed):
        
        Nx = params.get("Nx", 256)
        Ny = params.get("Ny", 256)
        xleng = params.get("xleng", 40)
        yleng = params.get("yleng", 40)
        T = params.get("T", 128)
        dt = params.get("dt", 0.035)
        Du = params.get("Du", 1.0)
        Dv = params.get("Dv", 0.01)
        k1 = params.get("k1", 5)
        k2 = params.get("k2", 11.0)
        ns = params.get("ns", 8)
        
        ou, ov, dx, dy = self.initialize_grid(
            Nx, Ny, xleng, yleng, k2, seed
        )

        MaxIter = int(T/dt)
        kk = self.compute_k_grid(Nx, Ny, dx, dy)
        
        sim_time = 0
        tlist, ulist, vlist = [sim_time], [ou.copy()], [ov.copy()]

        start = time.time()
        
        for it in tqdm(range(MaxIter)):
            sim_time += dt

            ff = ou + dt * (k1 * (ov - ou * ov / (1 + ov ** 2)))
            gg = ov + dt * (k2 - ov - 4 * ou * ov / (1 + ov ** 2))

            ou = self.solver_fft(dt, Du, kk, ff)
            ov = self.solver_fft(dt, Dv, kk, gg)

            if (it + 1)%(MaxIter//ns) == 0:
                tlist.append(sim_time)
                ulist.append(ou.copy())
                vlist.append(ov.copy())
        
        end = time.time() - start
        print(f"Generate time: {end:.4f} sec")
        
        results = {
            "parameters": {
                "Du":Du, "Dv":Dv, "k1":k1, "k2":k2, 
                "Nx":Nx, "Ny":Ny, 
                "xleng":xleng, "yleng":yleng,
                "T": T, "dt": dt
            },
            "tlist": tlist, 
            "ulist": ulist, 
            "vlist": vlist
        }

        return results