import pickle
import argparse
import os
import itertools

from solvers.fft_solver import run_2d_simulation



def parse_points(inputs):

    points = []
    for p in inputs:
        Dv, k1 = p.split(",")
        points.append((float(Dv), float(k1)))
    return points


def gen_dataset(args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    param_points = parse_points(args.points)
    
    idx = 0
    for (Dv, k1), seed in itertools.product(param_points, args.seeds):
        print(f"**Running: (Dv={Dv:.4f}, k1={k1}, seed={seed})")
        
        results = run_2d_simulation(
            Nx=args.Nx,
            Ny=args.Ny,
            xleng=args.xleng,
            yleng=args.yleng,
            T=args.T,
            dt=args.dt,
            Dv=Dv, 
            k1=k1, 
            seed=seed,
            ns=args.ns
        )
        
        filename = f"{idx}.pkl"
        save_path = os.path.join(args.save_dir, filename)
        
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        
        idx += 1
        
        
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--points", 
        nargs="+",
        default=["0.01,5", "0.04,1", "0.03,5", "0.04,7"]
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int, 
        default=[1,2,3,4,5]
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="data/dataset"
    )
    
    parser.add_argument("--Nx", type=int, default=256)
    parser.add_argument("--Ny", type=int, default=256)
    parser.add_argument("--xleng", type=float, default=40)
    parser.add_argument("--yleng", type=float, default=40)
    parser.add_argument("--T", type=float, default=128)
    parser.add_argument("--dt", type=float, default=0.035)
    parser.add_argument("--ns", type=int, default=8)
    
    return parser.parse_args()


def main():
    
    args = parse_args()
    gen_dataset(args)
    
    
    
if __name__=="__main__":
    main()