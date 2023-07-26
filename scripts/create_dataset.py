import argparse
from util import parse_json, setup_logging, load_policy
import logging
import numpy as np

from src.algorithms import evaluate_policy
from src.batch import new_batch, save_batch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=123456,
                    help="seed")
parser.add_argument("--batch_size", type=int, default=1,
                    help="size of the batch")
parser.add_argument("--episode_length_bound", type=int, default=200,
                    help="bound the length of each episode")
parser.add_argument('--output_file', type=str, default=None,
                    help='file to save batch')
parser.add_argument("--behavior_policy_config", default='scripts/configs/default_behavior_policy.json',
                    help="file with behaviour policy configuration to be evaluated.")
parser.add_argument('--env_id', default='TaxiAbsorbing-v0',
                    help='id of environment to run')
parser.add_argument("--discount_factor", type=float, default=0.99,
                    help="discount factor")
parser.add_argument("--dump_policies", action="store_true", help="dump policies to files")
parser.add_argument("--evaluate_behavior_policy", action="store_true", help="dump policies to files")
parser.add_argument("--log_level", default='ERROR', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help="choose log level")


def create_batch(seed, batch_size, env_id, behavior_policy_config,
                 output_file,
                 discount_factor, episode_length_bound,
                 dump_policies, log_level, evaluate_behavior_policy):
    np.random.seed(seed)
    setup_logging(log_level, env_id, behavior_policy_config["baseline_policy"])
    logging.info("Start", )
    behavior_policy, env = load_policy(behavior_policy_config["baseline_policy"], env_id, discount_factor,
                                       behavior_policy_config.get("softmax_temperature", None),
                                       behavior_policy_config.get("baseline_policy_path", None),
                                       dump_policies, seed)
    if evaluate_behavior_policy:
        baseline_value = evaluate_policy(env, behavior_policy, discount_factor, max_episode_size=episode_length_bound)
        logging.info('Value of the behavior policy: {}'.format(baseline_value))

    batch = new_batch(env, size=batch_size, policy=behavior_policy, episode_length_bound=episode_length_bound)
    if output_file is None:
        output_file = "/tmp/batch_{}_{}_{}.p".format(env_id, batch_size, seed)
    save_batch(batch, output_file)
    print("batch saved to {}".format(output_file))


if __name__ == '__main__':
    parsed_args = parser.parse_args()
    parsed_args.behavior_policy_config = parse_json(parsed_args.behavior_policy_config)
    create_batch(**vars(parsed_args))
