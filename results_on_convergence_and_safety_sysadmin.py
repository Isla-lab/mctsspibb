"""
This file is part of MCTS-SPIBB.

MCTS-SPIBB is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Foobar is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Foobar.
If not, see <https://www.gnu.org/licenses/>.

MCTS-SPIBB authors kindly ask to cite the paper "Scalable Safe Policy Improvement via Monte Carlo Tree Search" if the
code is used in other projects.
"""

import argparse
import pickle
from math import ceil, log

import numpy as np
from envs.sysadmin_generative import SysAdmin
from src.algorithms import policy_iteration
from src.batch import new_batch
from src.mcts import Mcts
from src.models import get_baseline_policy
from src.util import save_object, get_random_policy_and_env, compute_mask, get_configs_from_dicts, \
    parse_json, policy_evaluation
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--env_id', default='SysAdmin4-v0',
                    help='id of environment to run')
parser.add_argument("--n_machines", type=int, default=4,
                    help="number of machines")
parser.add_argument("--batch_size", type=int, default=5000,
                    help="size of the batch")
parser.add_argument("--discount_factor", type=float, default=0.95,
                    help="discount factor")
parser.add_argument("--episode_length_bound", type=int, default=200,
                    help="bound the length of each episode")
parser.add_argument("--budget", type=int, default=0,
                    help="seconds available to compute the policy")
parser.add_argument("--n_sims", type=int, default=100,
                    help="number of simulations for MCTS-SPIBB")
parser.add_argument("--behavior_policy_config", default='scripts/configs/default_behavior_policy.json',
                    help="file with behaviour policy configuration to be evaluated.")
parser.add_argument("--agents_config", default='scripts/configs/default_agents.json',
                    help="file with algorithms configurations to be evaluated.")


def policy_comparison(agents_config, env_id, n_machines, batch_size, discount_factor,
                      episode_length_bound, budget, behavior_policy_config, n_sims):
    behavior_policy, env = get_random_policy_and_env(behavior_policy_config["baseline_policy"], env_id, discount_factor,
                                                     behavior_policy_config.get("softmax_temperature", None),
                                                     behavior_policy_config.get("baseline_policy_path", None))

    save_object(env, 'out/SysAdmin/env_%s.pkl' % env_id)
    n_states = 2 ** n_machines
    n_actions = n_machines + 1
    n_wedge = 50
    nb_trajectories = 10000

    real_T = np.zeros((env.nS, env.nA, env.nS))
    R = np.zeros((env.nS, env.nA))
    for s in range(0, env.nS):
        for a in range(0, env.nA):
            for elem in env.P[s][a]:
                real_T[s][a][elem[1]] = elem[0]
                R[s][a] = elem[2]

    save_object(real_T, 'out/SysAdmin/real_T_%s.pkl' % env_id)
    save_object(R, 'out/SysAdmin/R_%s.pkl' % env_id)

    print('Policy comparison started:')
    # Parameters of comparison
    initial_state = 0
    batch = new_batch(env, size=nb_trajectories, policy=behavior_policy,
                      episode_length_bound=episode_length_bound)

    for config in agents_config:
        get_baseline_policy(batch,
                            env,
                            env_id,
                            behavior_policy,
                            representation=config["representation"],
                            safety_strategy=config["safety_strategy"],
                            structure_type=config.get("structure", None),
                            discount_factor=discount_factor,
                            config=config,
                            )

    with open('out/SysAdmin/baseline_%s.pkl' % env_id, 'rb') as inp:
        baseline = pickle.load(inp)

    with open('out/SysAdmin/mdp_%s.pkl' % env_id, 'rb') as inp:
        mdp = pickle.load(inp)

    print('1. Baseline: compute the performance')
    perf_baseline = policy_evaluation(baseline, R, real_T, discount_factor)[0][initial_state]

    print('2. SPIBB: compute the performance')

    q_values = policy_iteration(mdp, epsilon=0.01, max_iterations=episode_length_bound,
                                initial_policy=mdp.behavior_policy,
                                initial_q=mdp.behavior_policy.q_values)
    spibb = mdp.get_safe_policy(q_values)
    spibb = spibb.full_distribution()

    perf_spibb = policy_evaluation(spibb, R, real_T, discount_factor)[0][initial_state]

    print('3. MCTS-SPIBB: compute the performance')
    with open('out/SysAdmin/q/matrix_of_visits_%s.pkl' % env_id, 'rb') as inp:
        matrix_of_visits = pickle.load(inp)

    with open('out/SysAdmin/mle/MLE_T_%s.pkl' % env_id, 'rb') as inp:
        MLE_T = pickle.load(inp)

    # Parameters of mcts
    exp_constant = 5
    gamma = 0.85
    tree_depth = ceil(log(0.01, gamma))
    type_node = 'uniform'
    final_states = [10000000000]

    mask = compute_mask(matrix_of_visits, n_wedge)
    mcts = np.zeros((n_states, n_actions))
    for state in range(n_states):
        pib_MCTS = Mcts(gamma, n_states, n_actions, baseline, MLE_T, R, mask, q_values, 0,
                        'default', n_sims=n_sims, exploration_costant=exp_constant,
                        max_depth=tree_depth,
                        type_node=type_node,
                        states_to_sim=[state], final_states=final_states)
        pib_MCTS.fit()
        mcts[state] = pib_MCTS.pi[state]

    perf_mcts = policy_evaluation(mcts, R, real_T, discount_factor)[0][initial_state]

    print('Baseline: %s' % perf_baseline)
    print('SPIBB: %s' % perf_spibb)
    print('MCTS-SPIBB_n_sims = %s: %s' % (n_sims, perf_mcts))
    print('SPIBB - MCTS-SPIBB_n_sims = %s = %s' % (n_sims, perf_spibb - perf_mcts))

if __name__ == '__main__':
    parsed_args = parser.parse_args()
    parsed_args.agents_config = get_configs_from_dicts(parse_json(parsed_args.agents_config))
    parsed_args.behavior_policy_config = parse_json(parsed_args.behavior_policy_config)

    policy_comparison(**vars(parsed_args))
