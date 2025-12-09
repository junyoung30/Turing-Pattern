import matplotlib.pyplot as plt
import pickle
import argparse
import os

from solvers.fft_solver import run_2d_simulation



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--Dv", type=float, default=0.01)
    parser.add_argument("--k1", type=float, default=5)
    parser.add_argument("--Nx", type=int, default=256)
    parser.add_argument("--Ny", type=int, default=256)
    parser.add_argument("--xleng", type=float, default=40)
    parser.add_argument("--yleng", type=float, default=40)
    parser.add_argument("--T", type=float, default=128)
    parser.add_argument("--dt", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=1004)
    parser.add_argument("--ns", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="data/sim_test")
    
    return parser.parse_args()

def save_simulation(results, args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    filename = f"Dv_{args.Dv:.4f}_k1_{args.k1}_seed_{args.seed}.pkl"
    save_path = os.path.join(args.save_dir, filename)
    
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    
    return save_path


def show_pattern(results):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(results["vlist"][-1], cmap="jet")
    ax.axis("off")
    plt.show()


def main():
    
    args = parse_args()
    
    print(f"**Running: Dv={args.Dv:.4f}, k1={args.k1}, seed={args.seed}")
    results = run_2d_simulation(
        Nx=args.Nx, 
        Ny=args.Ny, 
        xleng=args.xleng, 
        yleng=args.yleng,
        T=args.T, 
        dt=args.dt, 
        Dv=args.Dv,
        k1=args.k1,
        seed=args.seed,
        ns=args.ns
    )
    
    show_pattern(results)
    save_simulation(results, args)
    
    
if __name__=="__main__":
    main()