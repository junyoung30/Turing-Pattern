import pickle
import argparse
import os
import itertools

from solvers.fft_solver import run_2d_simulation



def parse_points(inputs):

    points = []
    for p in inputs:
        parts = [x.strip() for x in p.split(",")]
        if len(parts) != 3:
            raise ValueError("Expected 'Dv,k1,label'")
            
        Dv, k1, label = parts
        points.append((float(Dv), float(k1), label))
    return points


def gen_dataset(args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    param_points = parse_points(args.points)
    
    idx = 0
    for (Dv, k1, label), seed in itertools.product(param_points, args.seeds):
        print(f"**Running: (label={label}, Dv={Dv:.4f}, k1={k1}, seed={seed})")
        
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
        results["pattern"] = {"label": label}
        
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
        default=["0.01,5,Sdot", "0.04,1,Ldot", "0.03,5,Line", "0.04,7,Mdot"]
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