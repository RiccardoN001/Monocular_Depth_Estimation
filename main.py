import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4) # This will not be used, since the learning rate is set differently for encoder and decoder
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=80)
    
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint")
    parser.add_argument("--ckpt_name", type=str, default="depth")
    parser.add_argument("--evaluate_every", type=int, default=2)
    parser.add_argument("--visualize_every", type=int, default=100)
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join("./DepthEstimationUnreal"))

    parser.add_argument("--is_train", type=bool, default=False)
    parser.add_argument("--ckpt_file", type=str, default="depth_60.pth")

    args = parser.parse_args()

    # --- initialize solver ---
    solver = Solver(args)

    if args.is_train:
        solver.fit()
    else:
        solver.test()

if __name__ == "__main__":
    main()
