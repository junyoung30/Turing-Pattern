from fft import run_2d_simulation
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Dv", type=float, default=0.01)
    parser.add_argument("--k1", type=float, default=5)
    parser.add_argument("--Nx", type=int, default=256)
    parser.add_argument("--Ny", type=int, default=256)
    parser.add_argument("--T", type=float, default=128)
    parser.add_argument("--dt", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=1004)
    parser.add_argument("--ns", type=int, default=8)
    
    args = parser.parse_args()
        
    results = run_2d_simulation(
        Nx=args.Nx, 
        Ny=args.Ny, 
        T=args.T, 
        dt=args.dt, 
        Dv=args.Dv,
        k1=args.k1,
        seed=args.seed, 
        ns=args.ns
    )
    
    plt.imshow(results['vlist'][-1], cmap='jet')
    plt.axis('off')
    plt.show()
    
    os.makedirs("results", exist_ok=True)
    save_path = f"results/Dv{args.Dv}_k1_{args.k1}_seed{args.seed}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
        
    print(f"Simulation complete. Saved to {save_path}")
    
if __name__=="__main__":
    main()