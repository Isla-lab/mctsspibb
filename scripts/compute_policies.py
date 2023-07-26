import argparse
import time
import logging
import numpy as np

from src.algorithms import value_iteration, evaluate_policy, policy_iteration
from src.batch import load_batch
from src.util import dicts_to_csv
from src.models import construct_model_from_batch
from util import load_policy, setup_logging, get_configs_from_dicts, parse_json

parser = argparse.ArgumentParser()
parser.add_argument("--agents_config", default='scripts/configs/default_agents.json',
                    help="file with algorithms configurations to be evaluated.")
parser.add_argument("--discount_factor", type=float, default=0.99,
                    help="discount factor")
parser.add_argument("--episode_length_bound", type=int, default=200,
                    help="bound the length of each episode")
parser.add_argument("--mdp_solver", default='value_iteration', choices=['value_iteration', 'policy_iteration'],
                    help="algorithm to compute optimal policy for estimated mdp")
parser.add_argument('--output_file', type=str, default='out/results.csv',
                    help='output file')
parser.add_argument('--batch_path', type=str, default='/tmp/batch.p',
                    help='path to batch file')
parser.add_argument('--policy_file', type=str, default=None,
                    help='path to batch file')

parser.add_argument("--behavior_policy_config", default='scripts/configs/default_behavior_policy.json',
                    help="file with behaviour policy configuration to be evaluated.")
parser.add_argument('--env_id', default='TaxiAbsorbing-v0',
                    help='id of environment to run')

parser.add_argument("--log_level", default='ERROR', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help="choose log level")
parser.add_argument("--dump_policies", action="store_true", help="dump policies to files")
parser.add_argument("--evaluate_policies", action="store_true")


def compute_policies(agents_config, batch_path, env_id, behavior_policy_config,
                     policy_file,
                     output_file,
                     discount_factor, episode_length_bound, mdp_solver,
                     dump_policies, evaluate_policies, log_level):
    setup_logging(log_level, env_id, behavior_policy_config["baseline_policy"])
    logging.info("Start", )
    behavior_policy, env = load_policy(behavior_policy_config["baseline_policy"], env_id, discount_factor,
                                       behavior_policy_config.get("softmax_temperature", None),
                                       behavior_policy_config.get("baseline_policy_path", None), dump_policies)

    baseline_value = evaluate_policy(env, behavior_policy, discount_factor, max_episode_size=episode_length_bound)
    logging.info('Value of the behavior policy: {}'.format(baseline_value))

    results = []
    batch = load_batch(batch_path)
    for config in agents_config:

        result = dict()
        start_time = time.perf_counter()
        mdp = construct_model_from_batch(batch,
                                         env,
                                         behavior_policy,
                                         representation=config["representation"],
                                         safety_strategy=config["safety_strategy"],
                                         structure_type=config.get("structure", None),
                                         discount_factor=discount_factor,
                                         config=config,
                                         )
        result["time_pre_processing"] = time.perf_counter() - start_time

        logging.info("Model: {} constructed in {} s".format(mdp, result["time_pre_processing"]))

        start_time = time.perf_counter()
        if mdp_solver == "value_iteration":
            if hasattr(behavior_policy, 'q_values'):
                q_values = value_iteration(mdp, epsilon=0.01, max_iterations=episode_length_bound,
                                           initial_q=behavior_policy.q_values)
            else:
                q_values = value_iteration(mdp, epsilon=0.01, max_iterations=episode_length_bound)
        elif mdp_solver == "policy_iteration":
            if hasattr(behavior_policy, 'q_values'):
                q_values = policy_iteration(mdp, epsilon=0.01, max_iterations=episode_length_bound,
                                            initial_policy=behavior_policy,
                                            initial_q=behavior_policy.q_values)
            else:
                q_values = policy_iteration(mdp, epsilon=0.01, max_iterations=episode_length_bound,
                                            initial_policy=behavior_policy)

        result["time_optimization"] = time.perf_counter() - start_time

        safe_policy = mdp.get_safe_policy(q_values)
        if dump_policies:
            if policy_file is None:
                # TODO save policy with unique name
                policy_file = '/tmp/policy.p'
            safe_policy.dump(policy_file)

        if evaluate_policies:
            value = evaluate_policy(env, safe_policy, discount_factor, max_episode_size=episode_length_bound)
            divergences = np.array(mdp.divergence_transition_function())
            bootstrapped = np.array(mdp.bootstrapped_state_action_pairs())
            non_bootstrapped = np.logical_not(bootstrapped)
            result.update({
                "batch_size": len(batch),
                "batch_path": batch_path,
                "E[s_0, pi]": value,
                "model": str(mdp),
                "baseline_policy_config": behavior_policy_config,
                "env": env_id,
                "baseline_value": baseline_value,
                "time_total": result["time_optimization"] + result["time_pre_processing"],
                "episode_length_bound": episode_length_bound,
                "error_transition_function_mean": divergences.mean(),
                "error_transition_function_std": divergences.std(),
                "bootstrapped_state_action_pairs": bootstrapped.sum(),
                "discount_factor": discount_factor,
                "horizon": episode_length_bound,
                "mdp_solver": mdp_solver,
                "agent_config": config,

            })
            result.update(mdp.structure_error())
            if bootstrapped.sum() > 0:
                result["mean_error_bootstrapped_state_action_pairs"] = divergences[bootstrapped].mean()
            else:
                result["mean_error_bootstrapped_state_action_pairs"] = 0
            if non_bootstrapped.sum() > 0:
                result["mean_error_non_bootstrapped_state_action_pairs"] = divergences[non_bootstrapped].mean()
            else:
                result["mean_error_non_bootstrapped_state_action_pairs"] = 0
            results.append(result)

    if evaluate_policies:
        dicts_to_csv(results, output_file)


if __name__ == '__main__':
    parsed_args = parser.parse_args()
    parsed_args.agents_config = get_configs_from_dicts(parse_json(parsed_args.agents_config))
    parsed_args.behavior_policy_config = parse_json(parsed_args.behavior_policy_config)
    compute_policies(**vars(parsed_args))
