#!/opt/miniconda3/envs/thesis/bin/python
import sys
import argparse
import os
import wandb

sys.path.append(os.path.abspath('/Users/justinband/repos/masters-thesis/'))
from simulator import Simulator

def str_to_bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def sweep_run():
    print("Performing Sweep")
    run = wandb.init(project='sweep-green-scheduling')
    config = run.config

    job_size = config.job_size
    alpha = config.alpha
    lr = config.lr

    sim = Simulator(['lfa-ql'],
                    job_size=job_size,
                    episodes=10000,
                    alpha=alpha,
                    lr=lr,
                    verbose=False,
                    normalized=True)
    sim.wandb_log = True
    sim.train()

    iterations = 10000
    sim.evaluate(iterations=iterations)

def get_sweep_config():
    sweep_config = {
        "method": "grid"
    }
    metric = {
        'name': 'eval/carbon/mean',
        'goal': 'minimize'
    }
    # parameters = {
    #     'job_size': {
    #         'values': [2, 5, 10, 15, 20]
    #     },
    #     'alpha': {
    #         'values': [1, 2, 4, 5, 8, 10, 12, 15]
    #     },
    #     'lr': {
    #         'values': [1e-3, 1e-4, 1e-5, 1e-6]
    #     }
    # }
    parameters = {
            'job_size': {
                'values': [20]
                # 'values': [2, 5, 10, 15, 20, 30]
            },
            'alpha': {
                'values': [2, 5, 8, 10, 15, 20]
            },
            'lr': {
                'values': [1e-5]
            }
        }
    sweep_config['parameters'] = parameters
    sweep_config['metric'] = metric
    return sweep_config

if __name__ == "__main__":
    # Defaults
    job_size = 10
    episodes = 1250
    alpha = 1
    iterations = 10000
    normalize = True
    lr = 1e-5

    existing_algs = ['ql',
                     'lfa-ql',
                     'run-only']

    parser = argparse.ArgumentParser(description="Run simulator with a specified algorithm.")
    parser.add_argument("algorithms", nargs="*", help=f"Algorithm to run. Options: {", ".join(existing_algs)}")
    parser.add_argument("-e", "--episodes", type=int, default=episodes, help="Number of episodes to train on")
    parser.add_argument("-j", "--job-size", type=int, default=job_size, help="Size of a job")
    parser.add_argument("-a", "--alpha", type=int, default=alpha, help="Tradeoff of how long we're willing to wait.")
    parser.add_argument("-n", "--normalize", type=str_to_bool, default=normalize, help="Uses normalized data in [0,1] when true, otherwise uses original data")
    parser.add_argument("-lr", "--learning_rate", type=float, default=lr, help='Sets the learning rate.')

    parser.add_argument("-eval", "--evaluate", type=str_to_bool, default=False, help="Evaluate saved models or not")
    parser.add_argument("-s", "--seed", type=int, help="Defines a seed. Useful for reproduction.")
    parser.add_argument("-v", "--verbose", type=str_to_bool, default=True, help="Sets whether training information should be displayed.")

    parser.add_argument("-sweep", "--sweep", type=str_to_bool, default=False, help="Perform pre-defined sweep")
    args = parser.parse_args()

    if args.sweep:
        sweep_config = get_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project="sweep-green-scheduling")
        wandb.agent(sweep_id, function=sweep_run)
    else:
        if not args.algorithms:
            raise argparse.ArgumentTypeError("Algorithm is required. See help for details.")
        for a in args.algorithms:
            if a not in existing_algs:
                raise argparse.ArgumentError(f"Algorithm {a} cannot be handled.")
            
        sim = Simulator(args.algorithms,
                        args.job_size, 
                        args.episodes,
                        args.alpha,
                        args.learning_rate,
                        args.verbose,
                        args.normalize,
                        args.seed)
        if args.evaluate:
            sim.evaluate(iterations=iterations)
        else:
            sim.train()
            sim.evaluate(iterations=iterations)
