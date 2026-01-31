# Description

This repo is the codebase for my 4-month thesis titled, "GreenQ-GS: Balancing Carbon and Latency in Green Scheduling Problems with Reinforcement Learning".

* Supervisor: Yevgeny Seldin
* Co-Supervisor: Raghavendra Selvan

## Thesis Abstract

This thesis presents CarbonQ-GS, a contextual reinforcement learning algorithm for green scheduling problems. Unlike throughput-focused schedulers, green scheduling focuses on reducing the carbon footprint of computational jobs by adjusting a job’s execution time. We formulate a Markov Decision Process (MDP) environment for a single job, where the job can be paused and resumed to exploit low-carbon periods. The MDP’s loss function is uniquely derived based on a carbon-latency trade-off and a user-defined completion deadline. CarbonQ-GS combines Q-learning and linear function approximation to balance this trade-off. Trained and evaluated on hourly carbon intensity data from Denmark, CarbonQGS outperforms a maximal-throughput baseline, achieving an average 4.45% reduction in carbon intensity by doubling the job’s minimum completion time, with greater reductions as flexibility increases, all while respecting the completion deadline. Although a gap remains between CarbonQ-GS and a carbon optimal scheduler, our environment and algorithm strike a practical and straightforward approach for balancing environmental impact and completion time, providing viable first steps in modelling and scheduling flexible jobs.

# Installation

To run the simulator, see and run `simulator/install_simulator.sh`. This script installs `simulator_cli.py`, making it an available command on the CLI.

Given that I am writing this README months after I've last ran this code there may be other required steps that I am missing.

# Running

After installation, you can run the simulator with any of the available arguments. I have hooked up the simulator to run with WandB, which I highly recommend for logging purposes.
