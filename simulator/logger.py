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
        "train/episode": e,
        "train/loss": loss,
        "train/carbon": carbon,
        "train/optimal_carbon": opt_carbon,
        "train/regret": regret
    })

def eval_log(data_type, stats):
    wandb.log({
        f"eval/{data_type}/mean": stats['mean'],
        f"eval/{data_type}/std": stats['std'],
        f"eval/{data_type}/base_mean": stats['base_mean'],
        f"eval/{data_type}/base_std": stats['base_std'],
        f"eval/{data_type}/base_diff_percent": stats['base_diff_percent']
    })

def log(name, val):
    wandb.log({name: val})

def log_image(img, name):
    wandb.log({
        name: wandb.Image(img)
    })

def get_name_from_config(alg_title, config):
    name = f"{alg_title}_j{config['job_size']}_a{config['alpha']}_lr{config['learning_rate']}_ep{config['episodes']}"
    return name