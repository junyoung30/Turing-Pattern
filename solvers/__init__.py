from .fft_numpy import FFTSolverNumpy
from .fft_jax import FFTSolverJax

SOLVER_REGISTRY = {
    "fft_numpy": FFTSolverNumpy,
    "fft_jax": FFTSolverJax,
}