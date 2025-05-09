import wandb

def init_logging(project, config, alg_title):
    name = get_name_from_config(alg_title, config)
    wandb.init(
        project=project,
        name=name,
        config=config
    )

def train_log(e, loss, carbon, opt_carbon, regret):
    wandb.log({
        "episode": e,
        "train_loss": loss,
        "carbon": carbon,
        "optimal_carbon": opt_carbon,
        "regret": regret
    })

def eval_log(data_type, mean, std, diff):
    wandb.log({
        f"{data_type}/mean": mean,
        f"{data_type}/std": std,
        f"{data_type}/baseline_diff": diff
    })

def get_name_from_config(alg_title, config):
    name = f"{alg_title}_j{config['job_size']}_a{config['alpha']}_lr{config['learning_rate']}_ep{config['episodes']}"
    return name