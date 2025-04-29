import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from algorithms.run_only import RunAgent

def save_model(model, model_name):
    model_dir = f"models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    
    # Dump model
    weights = model.get_weights()
    joblib.dump(weights, model_path)

def load_model(model, model_name):
    model_path = f"models/{model_name}.pkl"
    if os.path.exists(model_path):
        weights = joblib.load(model_path)
        model.load_weights(weights)
        print(f"Loaded saved {model_name} model")
        return model
    else:
        raise Exception(f"Saved {model_name} model not found")

def add_baseline_alg(env, models):
    agent = RunAgent(env)
    key = 'run-agent'
    value = {'title': 'Run-Only Agent', 'alg': agent}
    models.setdefault(key, value)
    return models

def plot_evaluation_results(loss_history, carbon_history, iterations):
    fig, axes = plt.subplots(2, 1)
    fig.suptitle(f"Model Evaluation over {iterations} iterations")

    def get_trend_line(data, window=50):
        window_size = min(window, len(data)//10)
        smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        return smoothed

    for alg, losses in loss_history.items():
        axes[0].plot(losses, label=alg, alpha=0.7)
        axes[0].set_title("Losses")
        axes[0].set_ylabel("Loss")
        axes[0].set_xlabel("Time")
        axes[0].legend()
        axes[0].grid(True)

    for alg, carbons in carbon_history.items():
        window = 100
        trend = get_trend_line(carbons, window)

        axes[1].plot(range(window-1, len(carbons)),
                        trend,
                        label=f'{alg}-trend',
                        linewidth=2,
                        zorder=2)
        axes[1].plot(carbons, label=alg, alpha=0.5, zorder=1)
        axes[1].set_title("Carbons")
        axes[1].set_ylabel("Carbon")
        axes[1].set_xlabel("Time")
        axes[1].legend()
        axes[1].grid(True)