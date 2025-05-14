import io
import os
import matplotlib
import numpy as np
if os.environ.get("WANDB_SWEEP") == "true":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import utils, logger

def run_eval(env, models):
    results = {}
    start_idx = env.get_random_index()

    for alg_i, (alg_name, alg_dict) in enumerate(models.items()):
        agent = alg_dict['alg']
        total_loss, action_history, intensity_history, state_history, loss_history, q_vals_history, total_carbon, carbon_alpha = agent.evaluate(start_idx)

        # plotting.plot_evaluation_results(
        #     actions=action_history,
        #     intensities=intensity_history,
        #     losses=loss_history,
        #     q_vals=q_vals_history,
        #     title=alg_dict['title']
        # )
        # print(f"[{alg_title}] Total loss: {total_loss:.2f}")
        # print(f"[{alg_title}] Total Carbon: {total_carbon:.5f}")
        # print(f"[{alg_title}] Job completed in {len(action_history)} hours")
        results[alg_name] = {
            'loss': np.round(total_loss, 2),
            'carbon': np.round(total_carbon, 5),
            'hours': len(action_history),
            'carbon_1/alpha': carbon_alpha
        }

    # baseline = 'run-agent'
    # for key in results:
    #     if key != baseline:
    #         carbon_diff = generic.calculate_diff(results[baseline]['carbon'], results[key]['carbon'])
    #         loss_diff = generic.calculate_diff(results[baseline]['loss'], results[key]['loss'])
    #         # time_diff = generic.calculate_diff(results[baseline]['hours'], results[key]['hours'])
    #         print(f'Loss difference from run-agent to {key} = {loss_diff}%')
    #         print(f'Carbon difference from run-agent to {key} = {carbon_diff}%')
    #         # print(f'Time difference from run-agent to {key} = {time_diff}')

    # plt.show()
    return results


def evaluate(env, iterations, models, config):
    print(f"Evaluating for {iterations} iterations")
    for i, (alg_name, dict) in enumerate(models.items()):        
        loaded_model = utils.load_model(dict['alg'], alg_name, config)
        models[alg_name]['alg'] = loaded_model

        logger.init_logging(project='test-green-scheduling',
                            config=config,
                            alg_title=alg_name
                            )
    utils.add_baseline_alg(env, models)
        
    results = []
    for i in range(iterations):
        if i % 1000 == 0:
            print(f"Iterations: {i}")
        eval = run_eval(env, models)
        results.append(eval)

    # Plotting
    alg_keys = results[0].keys()
    loss_history = {alg: [] for alg in alg_keys}
    carbon_history = {alg: [] for alg in alg_keys}
    time_history = {alg: [] for alg in alg_keys}
    carbon_alpha_history = {alg: [] for alg in alg_keys}

    for timestep in results:
        for alg, metrics in timestep.items():
            loss_history[alg].append(metrics['loss'])
            carbon_history[alg].append(metrics['carbon'])
            time_history[alg].append(metrics['hours'])
            carbon_alpha_history[alg].append(metrics['carbon_1/alpha'])

    calc_data_stats(loss_history, "loss")
    calc_data_stats(carbon_history, "carbon")
    calc_data_stats(time_history, "time")
    for key, data in carbon_alpha_history.items():
        if key != utils.get_baseline_key():
            logger.log("carbon_1/alpha mean", np.mean(data))

def calc_data_stats(data, data_title):
    print(f"--- Stats for {data_title} ---")
    baseline_key = utils.get_baseline_key()

    if data_title == 'carbon':
        calc_carbon_majority(data)

    for key, datum in data.items():
        mean = np.mean(datum)
        std = np.std(datum)
        print(f"[{key}] Mean {data_title}: {mean}")
        print(f"[{key}] Std {data_title}: {std}")
        if key != baseline_key:
            baseline_datum = np.array(data[baseline_key])
            new_datum = np.array(datum)

            # Differences
            diffs = utils.calc_diff(baseline_datum, new_datum)
            mean_diff = np.mean(diffs)
            mean_diff = np.round(mean_diff * 100, 2)
            print(f"[{key}] Mean {data_title} difference: {mean_diff}% {data_title} change")

            stat_dict = {
                "mean": mean,
                "std": std,
                "base_mean": np.mean(baseline_datum),
                "base_std": np.std(baseline_datum),
                "base_diff_percent": mean_diff
            }
            logger.eval_log(data_title, stat_dict)

def calc_carbon_majority(carbons):
    baseline_key = utils.get_baseline_key()
    baseline_carbons = carbons[baseline_key]

    for key, datum in carbons.items():
        if key != baseline_key:
            better_mask = np.array(datum) < baseline_carbons
            num_better = np.count_nonzero(better_mask)
            print(f'[{key}] #{num_better} better (lower) carbons out of {len(baseline_carbons)}')