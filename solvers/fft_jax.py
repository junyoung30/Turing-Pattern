import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import time

from .base import PatternSolver


class FFTSolverJax(PatternSolver):

    @staticmethod
    def initialize_grid(Nx, Ny, xleng, yleng, k2, seed):
        dx, dy = xleng / Nx, yleng / Ny

        x = jnp.linspace(0.5 * dx, xleng - 0.5 * dx, Nx)
        y = jnp.linspace(0.5 * dy, yleng - 0.5 * dy, Ny)
        xx, yy = jnp.meshgrid(x, y)

        key = jax.random.PRNGKey(seed)
        key_u, key_v = jax.random.split(key)
        noise_u = jax.random.uniform(key_u, shape=xx.shape, minval=-1.0, maxval=1.0)
        noise_v = jax.random.uniform(key_v, shape=xx.shape, minval=-1.0, maxval=1.0)

        iu = 1.0 + (k2**2) / 25.0 + 0.2 * noise_u
        iv = k2 / 5.0 + 0.2 * noise_v

        return iu, iv, dx, dy

    @staticmethod
    def solver_fft(dt, D, kk, bb):
        coef = 1.0 / (1.0 + dt * D * kk)
        bb_hat = jnp.fft.fft2(bb)
        nv_hat = coef * bb_hat
        return jnp.fft.ifft2(nv_hat).real
    
    @staticmethod
    def compute_k_grid(Nx, Ny, dx, dy):
        kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, dx)
        ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, dy)
        kx, ky = jnp.meshgrid(kx, ky)
        return kx**2 + ky**2

    @staticmethod
    def step_fn(carry, _):
        ou, ov, params = carry
        dt, Du, Dv, k1, k2, kk = params

        ff = ou + dt * (k1 * (ov - ou * ov / (1.0 + ov**2)))
        gg = ov + dt * (k2 - ov - 4.0 * ou * ov / (1.0 + ov**2))

        ou = FFTSolverJax.solver_fft(dt, Du, kk, ff)
        ov = FFTSolverJax.solver_fft(dt, Dv, kk, gg)

        return (ou, ov, params), None


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

        MaxIter = int(T / dt)

        ou, ov, dx, dy = self.initialize_grid(
            Nx, Ny, xleng, yleng, k2, seed
        )

        kk = self.compute_k_grid(Nx, Ny, dx, dy)

        params_jax = (dt, Du, Dv, k1, k2, kk)
        carry_init = (ou, ov, params_jax)
        
        start = time.time()
        
        (ou, ov, _), _ = lax.scan(
            self.step_fn,
            carry_init,
            xs=None,
            length=MaxIter
        )
        
        ou.block_until_ready()
        
        end = time.time() - start
        print(f"Generate time: {end:.4f} sec")

        ou = np.array(ou)
        ov = np.array(ov)
        
        results = {
            "parameters": {
                "Du": Du, "Dv": Dv, "k1": k1, "k2": k2,
                "Nx": Nx, "Ny": Ny,
                "xleng": xleng, "yleng": yleng,
                "T": T, "dt": dt,
            },
            "tlist": None,
            "ulist": [ou],
            "vlist": [ov],
        }
        
        return results