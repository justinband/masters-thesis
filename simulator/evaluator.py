import numpy as np
import matplotlib.pyplot as plt
from . import utils

def run_eval(env, models):
    results = {}
    ##### FIXME: THIS SHOULD BE REMOVED
    # start_idx = 6500
    #####
    start_idx = env.get_random_index()

    for alg_i, (alg_name, alg_dict) in enumerate(models.items()):
        agent = alg_dict['alg']
        total_loss, action_history, intensity_history, state_history, loss_history, q_vals_history, total_carbon = agent.evaluate(start_idx)

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
            'hours': len(action_history)
        }

    # print("--- Results ---")
    # pprint.pprint(results)

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


def evaluate(env, iterations, models):
    for i, (alg_name, dict) in enumerate(models.items()):
        loaded_model = utils.load_model(dict['alg'], alg_name)
        models[alg_name]['alg'] = loaded_model

    utils.add_baseline_alg(env, models)
        
    results = []
    for i in range(iterations):
        if i % 100 == 0:
            print(f"Iterations: {i}")
        eval = run_eval(env, models)
        results.append(eval)

    # Plotting
    alg_keys = results[0].keys()
    loss_history = {alg: [] for alg in alg_keys}
    carbon_history = {alg: [] for alg in alg_keys}
    time_history = {alg: [] for alg in alg_keys}

    for timestep in results:
        for alg, metrics in timestep.items():
            loss_history[alg].append(metrics['loss'])
            carbon_history[alg].append(metrics['carbon'])
            time_history[alg].append(metrics['hours'])

    utils.plot_evaluation_results(loss_history, carbon_history, iterations)
    plt.show()
    