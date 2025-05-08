import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from algorithms.run_only import RunAgent

def get_model_filename(model_name, info):
    js = info['job_size']
    alpha = info['alpha']
    lr = info['learning_rate']
    eps = info['episodes']
    name = f"{model_name}_j{js}_a{alpha}_lr{lr}_ep{eps}"
    return name 

def save_model(model, model_name, info):
    model_dir = f"models"
    os.makedirs(model_dir, exist_ok=True)

    name = get_model_filename(model_name, info)
    model_path = os.path.join(model_dir, f"{name}.pkl")
    
    # Dump model
    weights = model.get_weights()
    joblib.dump(weights, model_path)

def load_model(model, model_name, config=None):
    if config == None:
        name = model_name
    else:
        name = get_model_filename(model_name, config)

    model_dir = f"models"
    model_path = os.path.join(model_dir, f"{name}.pkl")

    if os.path.exists(model_path):
        weights = joblib.load(model_path)
        model.load_weights(weights)
        print(f"Loaded saved {model_name} model ({name})")
        return model
    else:
        raise Exception(f"Saved {model_name} model not found")

def add_baseline_alg(env, models):
    agent = RunAgent(env)
    key = 'run-agent'
    value = {'title': 'Run-Only Agent', 'alg': agent}
    models.setdefault(key, value)
    return models

def get_baseline_key():
    return 'run-agent'

def calc_diff(old_val, new_val):
    diff = (new_val - old_val) / old_val
    return np.round(diff, 3)

def plot_evaluation_results(loss_history, carbon_history, iterations):
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)
    fig.suptitle(f"Model Evaluation over {iterations} iterations")

    for alg, losses in loss_history.items():
        avg_loss = np.mean(losses)
        line, = axes[0].plot(losses, label=alg, alpha=0.3, zorder=1)
        # axes[0].plot(losses, label=alg, alpha=0.7)
        axes[0].axhline(avg_loss, label=f'Average {alg}', color=line.get_color(), linestyle='--')
        
           
    axes[0].set_title("Losses")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    for alg, carbons in carbon_history.items():
        avg_carbon = np.mean(carbons)
        line, = axes[1].plot(carbons, label=alg, alpha=0.3, zorder=1)
        # axes[1].plot(carbons, label=alg, alpha=0.7)
        axes[1].axhline(avg_carbon, label=f'Average {alg}', color=line.get_color(), linestyle='--')
        
    axes[1].legend()
    axes[1].set_title("Carbons")
    axes[1].set_ylabel("Carbon")
    axes[1].set_xlabel("Time")
    