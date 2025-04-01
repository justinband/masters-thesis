#!/opt/miniconda3/envs/thesis/bin/python
import sys
import argparse
import os

sys.path.append(os.path.abspath('/Users/justinband/repos/masters-thesis/'))
from simulator import Simulator

def str_to_bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    # Defaults
    job_size = 10
    latency = 1
    episodes = 100000

    existing_algs = ['ql', 'iql', 'linq']

    parser = argparse.ArgumentParser(description="Run simulator with a specified algorithm.")
    parser.add_argument("algorithms", nargs="*", help=f"Algorithm to run. Options: {", ".join(existing_algs)}")
    parser.add_argument("-e", "--episodes", type=int, default=episodes, help="Number of episodes to train on")
    parser.add_argument("-j", "--job-size", type=int, default=job_size, help="Size of a job")
    parser.add_argument("-l", "--latency", type=int, default=latency, help="Amount of latency we're willing to incur. Latency=0 means no latency. Latency=1 effectively doubles runtime")
    parser.add_argument("-i", "--iterations", type=int, default=1,help="Number of iterations to run training over. Results are averaged.")

    parser.add_argument("-s", "--seed", type=int, help="Defines a seed. Useful for reproduction.")
    parser.add_argument("-v", "--verbose", type=str_to_bool, default=True, help="Sets whether training information should be displayed.")

    args = parser.parse_args()

    assert args.algorithms != [], argparse.ArgumentTypeError("Algorithm is required. See help for details.")
    for a in args.algorithms:
        assert a in existing_algs, argparse.ArgumentError(f"Algorithm {a} cannot be handled.")
            
    sim = Simulator(args.algorithms,
                    args.job_size, 
                    args.episodes,
                    args.latency,
                    args.iterations,
                    args.verbose,
                    args.seed)
    sim.train()
    sim.plot_losses()
    sim.plot_regret()
    sim.plot_latency()
    