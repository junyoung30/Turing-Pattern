import os
import time
import argparse

from ml.trainer import prepare_data, train_model, save_results
from ml.utils import set_global_seed, show_results


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--num_conv", type=int, default=6)

    parser.add_argument("--aug", type=int, default=1)  # 1=True, 0=False
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)

    parser.add_argument("--data_seed", type=int, default=1004)
    parser.add_argument("--weight_seed", type=int, default=2025)

    return parser.parse_args()


def main():
    
    args = parse_args()
    
    DirData = "./data/dataset"
    SaveFolder = "./outputs"
    
    set_global_seed(1234)
    
    X_train, X_test, y_train, y_test = prepare_data(DirData, args.data_seed)
    
    model, history = train_model(
        X_train, 
        X_test, 
        y_train, 
        y_test,
        NumBlocks=args.num_blocks,
        NumConv=args.num_conv,
        NumDense=None,
        WeightSeed=args.weight_seed,
        AugFLAG=bool(args.aug),
        LR=args.lr,
        BS=args.bs,
        EPOCHS=args.epochs
    )
    
    show_results(history)
    
    model_name = (
        f"_CP{args.num_blocks}"
        f"_AUG{args.aug}"
        f"_conv{args.num_conv}"
        f"_weight{args.weight_seed:04d}"
    )
    
    save_results(
        model, history, SaveFolder, model_name
    )
    
if __name__=="__main__":
    main()