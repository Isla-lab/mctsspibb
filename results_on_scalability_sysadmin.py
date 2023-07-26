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
import gc
import pickle
import time
from math import ceil, log

import numpy as np

from envs.sysadmin_generative import SysAdmin
from src.algorithms import policy_iteration_with_budget, policy_iteration
from src.batch import new_batch
from src.mcts import Mcts
from src.mcts_ext import Mcts_ext
from src.models import get_baseline_policy, get_data
from src import spibb_utils, spibb, modelTransitions
from src.util import new_episode_with_generative_baseline, save_object, compute_compressed_mask, generative_baseline, \
    get_random_policy_and_env, compute_mask, get_configs_from_dicts, parse_json, load_env, policy_evaluation, \
    load_env_and_policies_sysadmin

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
parser.add_argument("--n_wedge", type=int, default=5,
                    help="N wedge")
parser.add_argument("--episode_length_bound", type=int, default=200,
                    help="bound the length of each episode")
parser.add_argument("--budget", type=int, default=0,
                    help="seconds available to compute the policy")
parser.add_argument("--n_runs", type=int, default=20,
                    help="number of runs for policy comparison")
parser.add_argument("--n_steps", type=int, default=20,
                    help="number of steps for each run")
parser.add_argument("--n_iterations", type=int, default=20,
                    help="number of iterations with different batch of data")
parser.add_argument("--n_sims", type=int, default=100,
                    help="number of simulations for MCTS-SPIBB")
parser.add_argument("--behavior_policy_config", default='scripts/configs/default_behavior_policy.json',
                    help="file with behaviour policy configuration to be evaluated.")
parser.add_argument("--agents_config", default='scripts/configs/default_agents.json',
                    help="file with algorithms configurations to be evaluated.")


def policy_comparison_extended(agents_config, env_id, n_machines, batch_size, discount_factor,
                               n_wedge, episode_length_bound, budget, n_runs, n_steps, n_iterations, n_sims,
                               behavior_policy_config):
    print('Create the environment')
    env = SysAdmin(n_machines)
    print('Environment is created')
    print('###########################################################################################################')

    for it in range(n_iterations):
        print('Iteration %s' % it)
        print('Compute a new batch')
        batch = list()
        for _ in range(batch_size):
            episode = new_episode_with_generative_baseline(env, episode_length_bound)
            batch.append(episode)
        save_object(batch, 'out/SysAdmin/batch/batch_%s_%sm_it_%s.pkl' % (env_id, n_machines, it))
        with open('out/SysAdmin/batch/batch_%s_%sm_it_%s.pkl' % (env_id, n_machines, it), 'rb') as inp:
            batch = pickle.load(inp)
        print('Batch is computed')
        print('#######################################################################################################')

        print('Create MLE T from batch')
        q = dict()
        for episode in batch:
            for (state, action, reward, next_state) in episode:
                if state not in q:
                    q[state] = dict()
                if action not in q[state]:
                    q[state][action] = dict()
                if next_state not in q[state][action]:
                    q[state][action][next_state] = 1
                else:
                    q[state][action][next_state] += 1

        save_object(q, 'out/SysAdmin/q/q_%s_%sm_it_%s.pkl' % (env_id, n_machines, it))
        with open('out/SysAdmin/q/q_%s_%sm_it_%s.pkl' % (env_id, n_machines, it), 'rb') as inp:
            q = pickle.load(inp)

        MLE_T = dict()
        ostates = np.array(list(q.keys()))
        ostates = np.sort(ostates)
        for state in ostates:
            MLE_T[state] = dict()
            oactions = np.array(list(q[state].keys()))
            oactions = np.sort(oactions)
            for action in oactions:
                MLE_T[state][action] = dict()
                sum_of_visits = 0
                onextstate = np.array(list(q[state][action].keys()))
                onextstate = np.sort(onextstate)
                sum_of_visits = sum(list(q[state][action].values()))
                for next_state in onextstate:
                    MLE_T[state][action][next_state] = (q[state][action][next_state] / sum_of_visits)

        for state in MLE_T.keys():
            for action in MLE_T[state].keys():
                list_ns = MLE_T[state][action].keys()
                sum_p_dead = 0
                na = 0
                for ns in list_ns:
                    if ns not in MLE_T.keys():
                        sum_p_dead += MLE_T[state][action][ns]
                        MLE_T[state][action][ns] = 0
                    else:
                        na += 1
                for ns in list_ns:
                    if ns in MLE_T.keys():
                        MLE_T[state][action][ns] += sum_p_dead / na
                        MLE_T[state][action][ns] = MLE_T[state][action][ns]

        for state in MLE_T.keys():
            for action in MLE_T[state].keys():
                p = np.sum(list(MLE_T[state][action].values()))
                for ns in MLE_T[state][action].keys():
                    if p != 0:
                        MLE_T[state][action][ns] /= p
                    else:
                        MLE_T[state][action][ns] = 0

        save_object(MLE_T, 'out/SysAdmin/mle/MLE_T_%s_%sm_it_%s.pkl' % (env_id, n_machines, it))
        with open('out/SysAdmin/mle/MLE_T_%s_%sm_it_%s.pkl' % (env_id, n_machines, it), 'rb') as inp:
            MLE_T = pickle.load(inp)

        n = dict()
        for episode in batch:
            for (state, action, reward, next_state) in episode:
                if state not in n:
                    n[state] = dict()
                if action not in n[state]:
                    n[state][action] = 1
                else:
                    n[state][action] += 1

        mask = compute_compressed_mask(n, n_wedge)
        save_object(mask, 'out/SysAdmin/mask/mask_%s_%sm_it_%s.pkl' % (env_id, n_machines, it))

        with open('out/SysAdmin/mask/mask_%s_%sm_it_%s.pkl' % (env_id, n_machines, it), 'rb') as inp:
            mask = pickle.load(inp)

        print('MLE T is created')
        print('#######################################################################################################')

        print('Policy comparison started:')
        print('1. Baseline: compute the performance')
        list_discounted_return_baseline = []
        for _ in range(n_runs):
            state = env.reset()
            array_of_visited_states = []
            array_of_selected_actions = []
            list_rewards_baseline = []
            for step in range(n_steps):
                array_of_visited_states.append(state)
                action = generative_baseline(env, state)
                array_of_selected_actions.append(action)
                state, reward, _ = env.step(state, action)
                list_rewards_baseline.append(reward)

            discounted_return_baseline = 0

            for step in range(n_steps):
                discounted_return_baseline += list_rewards_baseline[step] * pow(discount_factor, step)

            list_discounted_return_baseline.append(discounted_return_baseline)

            print('States: %s' % array_of_visited_states)
            print('Actions: %s' % array_of_selected_actions)
            print('List of Rewards: %s' % list_rewards_baseline)

        print('List of Discounted Return %s' % list_discounted_return_baseline)
        save_object(list_discounted_return_baseline,
                    'out/SysAdmin/results/list_discounted_return_baseline_%s_%sm_it_%s.pkl' % (env_id, n_machines, it))
        print('Average of Discounted Return %s' % np.mean(list_discounted_return_baseline))

        print('2. MCTS-SPIBB: compute the performance')
        # Parameters of mcts
        exp_constant = 5
        gamma = 0.85
        tree_depth = ceil(log(0.01, gamma))
        final_states = [10000000000]

        budget_per_step = budget / n_steps

        list_discounted_return_mcts_spibb = []
        for run in range(n_runs):
            print('Run: %s' % run)
            state = env.reset()
            array_of_visited_states = []
            array_of_selected_actions = []
            list_rewards_mcts_spibb = []
            for step in range(n_steps):
                if state in MLE_T.keys():
                    pib_mcts = Mcts_ext(gamma, env.n_states, env.n_actions, MLE_T, env, mask, budget_per_step, n_sims,
                                        exploration_costant=exp_constant, max_depth=tree_depth,
                                        states_to_sim=[state], final_states=final_states)
                    pib_mcts.fit()
                    action = np.argmax(pib_mcts.pi[state])
                else:
                    action = generative_baseline(env, state)

                array_of_visited_states.append(state)
                array_of_selected_actions.append(action)
                state, reward, _ = env.step(state, action)
                list_rewards_mcts_spibb.append(reward)

            discounted_return_mcts_spibb = 0
            for step in range(n_steps):
                discounted_return_mcts_spibb += list_rewards_mcts_spibb[step] * pow(discount_factor, step)

            list_discounted_return_mcts_spibb.append(discounted_return_mcts_spibb)

            print('States: %s' % array_of_visited_states)
            print('Actions: %s' % array_of_selected_actions)
            print('List of Rewards: %s' % list_rewards_mcts_spibb)

        print('List of Discounted Return %s' % list_discounted_return_mcts_spibb)
        save_object(list_discounted_return_mcts_spibb,
                    'out/SysAdmin/results/list_discounted_return_mcts_spibb_%s_%s_budget_%s_it_%s.pkl' % (
                        env_id, n_machines, budget, it))
        print('Average of Discounted Return %s' % np.mean(list_discounted_return_mcts_spibb))

        print('Policy comparison finished')


def policy_comparison(agents_config, env_id, n_machines, batch_size, discount_factor,
                      n_wedge, episode_length_bound, budget, n_runs, n_steps, behavior_policy_config, n_iterations,
                      n_sims):
    behavior_policy, env = get_random_policy_and_env(behavior_policy_config["baseline_policy"], env_id, discount_factor,
                                                     behavior_policy_config.get("softmax_temperature", None),
                                                     behavior_policy_config.get("baseline_policy_path", None))

    n_states = 2 ** n_machines
    n_actions = n_machines + 1

    print('Baseline policy computation')
    batch = new_batch(env, size=batch_size, policy=behavior_policy, episode_length_bound=episode_length_bound)

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
    del batch
    del behavior_policy
    gc.collect()

    with open('out/SysAdmin/baseline_%s.pkl' % env_id, 'rb') as inp:
        baseline = pickle.load(inp)

    with open('out/SysAdmin/env_%s.pkl' % env_id, 'rb') as inp:
        env = pickle.load(inp)

    with open('out/SysAdmin/mdp_%s.pkl' % env_id, 'rb') as inp:
        mdp = pickle.load(inp)

    real_T = np.zeros((env.nS, env.nA, env.nS))
    R = np.zeros((env.nS, env.nA))
    for s in range(0, env.nS):
        for a in range(0, env.nA):
            for elem in env.P[s][a]:
                real_T[s][a][elem[1]] = elem[0]
                R[s][a] = elem[2]

    del env
    gc.collect()

    save_object(real_T, 'out/SysAdmin/real_T_%s.pkl' % env_id)
    save_object(R, 'out/SysAdmin/R_%s.pkl' % env_id)
    for it in range(n_iterations):
        print('Iteration %s' % it)
        print('Policy comparison started:')
        # Parameters of comparison
        initial_state = 0

        print('1. Baseline: compute the performance')
        list_discounted_return_baseline = []
        for _ in range(n_runs):
            state = initial_state
            array_of_visited_states = []
            array_of_selected_actions = []
            list_rewards_baseline = []
            for step in range(n_steps):
                array_of_visited_states.append(state)
                action = np.random.choice(baseline.shape[1], p=baseline[state])
                array_of_selected_actions.append(action)
                list_rewards_baseline.append(R[state][action])
                probs = real_T[state][action]
                state = np.random.choice(probs.shape[0], p=probs)

            discounted_return_baseline = 0

            for step in range(n_steps):
                discounted_return_baseline += list_rewards_baseline[step] * pow(discount_factor, step)

            list_discounted_return_baseline.append(discounted_return_baseline)

            print('States: %s' % array_of_visited_states)
            print('Actions: %s' % array_of_selected_actions)
            print('List of Rewards: %s' % list_rewards_baseline)

        print('List of Discounted Return %s' % list_discounted_return_baseline)
        save_object(list_discounted_return_baseline,
                    'out/SysAdmin/results/list_discounted_return_baseline_%s_it_%s.pkl' % (env_id, it))
        print('Average of Discounted Return %s' % np.mean(list_discounted_return_baseline))

        print('2. SPIBB: compute the performance')

        if budget != 0:
            q_values = policy_iteration_with_budget(mdp, budget, epsilon=0.01, max_iterations=episode_length_bound,
                                                    initial_policy=mdp.behavior_policy,
                                                    initial_q=mdp.behavior_policy.q_values)
        else:
            q_values = policy_iteration(mdp, epsilon=0.01, max_iterations=episode_length_bound,
                                        initial_policy=mdp.behavior_policy,
                                        initial_q=mdp.behavior_policy.q_values)
        spibb = mdp.get_safe_policy(q_values)

        del mdp
        gc.collect()

        save_object(spibb.full_distribution(),
                    'out/SysAdmin/SPIBB_policy_%s_budget_%s_it_%s.pkl' % (env_id, budget, it))
        print('SPIBB: policy is saved')

        del spibb
        gc.collect()

        print('SPIBB: compute the performance')

        with open('out/SysAdmin/SPIBB_policy_%s_budget_%s_it_%s.pkl' % (env_id, budget, it), 'rb') as inp:
            spibb = pickle.load(inp)

        list_discounted_return_spibb = []
        for _ in range(n_runs):
            state = initial_state
            array_of_visited_states = []
            array_of_selected_actions = []
            list_rewards_spibb = []
            for step in range(n_steps):
                array_of_visited_states.append(state)
                action = np.argmax(spibb[state])
                array_of_selected_actions.append(action)
                list_rewards_spibb.append(R[state][action])
                probs = real_T[state][action]
                state = np.random.choice(probs.shape[0], p=probs)

            discounted_return_spibb = 0

            for step in range(n_steps):
                discounted_return_spibb += list_rewards_spibb[step] * pow(discount_factor, step)

            list_discounted_return_spibb.append(discounted_return_spibb)

            print('States: %s' % array_of_visited_states)
            print('Actions: %s' % array_of_selected_actions)
            print('List of Rewards: %s' % list_rewards_spibb)

        print('List of Discounted Return %s' % list_discounted_return_spibb)
        save_object(list_discounted_return_spibb,
                    'out/SysAdmin/results/list_discounted_return_spibb_%s_budget_%s_it_%s.pkl' % (env_id, budget, it))
        print('Average of Discounted Return %s' % np.mean(list_discounted_return_spibb))

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
        budget_per_step = budget / n_steps

        mask = compute_mask(matrix_of_visits, n_wedge)
        list_discounted_return_mcts_spibb = []
        for run in range(n_runs):
            q_values = np.zeros((n_states, n_actions))
            state = initial_state
            array_of_visited_states = []
            array_of_selected_actions = []
            list_rewards_mcts_spibb = []
            for step in range(n_steps):
                pib_MCTS = Mcts(gamma, n_states, n_actions, baseline, MLE_T, R, mask, q_values, budget_per_step,
                                'default', n_sims=n_sims, exploration_costant=exp_constant,
                                max_depth=tree_depth,
                                type_node=type_node,
                                states_to_sim=[state], final_states=final_states)
                pib_MCTS.fit()
                array_of_visited_states.append(state)
                action = np.argmax(pib_MCTS.pi[state])
                array_of_selected_actions.append(action)
                q_values[state] = pib_MCTS.q_values[state]
                list_rewards_mcts_spibb.append(R[state][action])
                probs = real_T[state][action]
                state = np.random.choice(probs.shape[0], p=probs)

            discounted_return_mcts_spibb = 0
            for step in range(n_steps):
                discounted_return_mcts_spibb += list_rewards_mcts_spibb[step] * pow(discount_factor, step)

            list_discounted_return_mcts_spibb.append(discounted_return_mcts_spibb)

            print('States: %s' % array_of_visited_states)
            print('Actions: %s' % array_of_selected_actions)
            print('List of Rewards: %s' % list_rewards_mcts_spibb)

        print('List of Discounted Return %s' % list_discounted_return_mcts_spibb)
        save_object(list_discounted_return_mcts_spibb,
                    'out/SysAdmin/results/list_discounted_return_mcts_spibb_%s_budget_%s_it_%s.pkl' % (
                        env_id, budget, it))
        print('Average of Discounted Return %s' % np.mean(list_discounted_return_mcts_spibb))


def time_comparison(agents_config, env_id, n_machines, batch_size, discount_factor,
                    n_wedge, episode_length_bound, budget, n_runs, n_steps, behavior_policy_config, n_iterations,
                    n_sims):
    number_of_machines = [4, 7, 10]
    spibb_time = []
    for m in number_of_machines:
        print('n_machines = %s' % m)
        n_states = 2 ** m
        n_actions = m + 1
        real_T, R, optimal, baseline, garnet, final_states, path = load_env_and_policies_sysadmin(m, discount_factor)
        trajectories, batch_traj = spibb_utils.generate_batch(10000, garnet, baseline)
        model = modelTransitions.ModelTransitions(batch_traj, n_states, n_actions)
        mask, q = spibb.compute_mask_N_wedge(n_states, n_actions, n_wedge, batch_traj)
        seed = 1
        np.random.seed(seed)
        start = time.time()
        pib_SPIBB = spibb.spibb(discount_factor, n_states, n_actions, baseline, mask, model.transitions, R,
                                'Pi_b_SPIBB')
        pib_SPIBB.fit()
        end = time.time()
        spibb_time.append(end - start)
        print(end - start)

    save_object(spibb_time, 'out/SysAdmin/results/spibb_computational_time.pkl')
    print('SPIBB computational time for machines %s' % number_of_machines)
    print(spibb_time)

    number_of_machines = [4, 7, 10, 12]
    spibb_dp_times = []
    for m in number_of_machines:
        print('n_machines = %s' % m)
        behavior_policy, env = get_random_policy_and_env(behavior_policy_config["baseline_policy"], env_id,
                                                         discount_factor,
                                                         behavior_policy_config.get("softmax_temperature", None),
                                                         behavior_policy_config.get("baseline_policy_path", None))
        print('Baseline policy computation')
        batch = new_batch(env, size=batch_size, policy=behavior_policy, episode_length_bound=episode_length_bound)

        for config in agents_config:
            baseline, mdp = get_data(batch,
                                     env,
                                     env_id,
                                     behavior_policy,
                                     representation=config["representation"],
                                     safety_strategy=config["safety_strategy"],
                                     structure_type=config.get("structure", None),
                                     discount_factor=discount_factor,
                                     config=config,
                                     )

        real_T = np.zeros((env.nS, env.nA, env.nS))
        R = np.zeros((env.nS, env.nA))
        for s in range(0, env.nS):
            for a in range(0, env.nA):
                for elem in env.P[s][a]:
                    real_T[s][a][elem[1]] = elem[0]
                    R[s][a] = elem[2]

        start = time.time()
        q_values = policy_iteration(mdp, epsilon=0.01, max_iterations=episode_length_bound,
                                    initial_policy=mdp.behavior_policy,
                                    initial_q=mdp.behavior_policy.q_values)
        spibb_dp = mdp.get_safe_policy(q_values)
        end = time.time()
        spibb_dp_times.append(end - start)
        print(end - start)

    save_object(spibb_dp_times, 'out/SysAdmin/results/spibb_dp_computational_time.pkl')
    print('SPIBB_DP computational time for machines %s' % number_of_machines)
    print(spibb_dp_times)

    number_of_machines = [4, 7, 10, 12, 20, 35]
    n_sims = [100, 4000, 10000]
    mcts_times = dict()
    for n_sim in n_sims:
        mcts_times[n_sim] = []
        for m in number_of_machines:
            print('n_sims = %s, n_machines = %s' % (n_sim, m))
            env = SysAdmin(m)

            batch = list()
            for _ in range(batch_size):
                episode = new_episode_with_generative_baseline(env, episode_length_bound)
                batch.append(episode)

            q = dict()
            for episode in batch:
                for (state, action, reward, next_state) in episode:
                    if state not in q:
                        q[state] = dict()
                    if action not in q[state]:
                        q[state][action] = dict()
                    if next_state not in q[state][action]:
                        q[state][action][next_state] = 1
                    else:
                        q[state][action][next_state] += 1

            MLE_T = dict()
            ostates = np.array(list(q.keys()))
            ostates = np.sort(ostates)
            for state in ostates:
                MLE_T[state] = dict()
                oactions = np.array(list(q[state].keys()))
                oactions = np.sort(oactions)
                for action in oactions:
                    MLE_T[state][action] = dict()
                    sum_of_visits = 0
                    onextstate = np.array(list(q[state][action].keys()))
                    onextstate = np.sort(onextstate)
                    sum_of_visits = sum(list(q[state][action].values()))
                    for next_state in onextstate:
                        MLE_T[state][action][next_state] = (q[state][action][next_state] / sum_of_visits)

            for state in MLE_T.keys():
                for action in MLE_T[state].keys():
                    list_ns = MLE_T[state][action].keys()
                    sum_p_dead = 0
                    na = 0
                    for ns in list_ns:
                        if ns not in MLE_T.keys():
                            sum_p_dead += MLE_T[state][action][ns]
                            MLE_T[state][action][ns] = 0
                        else:
                            na += 1
                    for ns in list_ns:
                        if ns in MLE_T.keys():
                            MLE_T[state][action][ns] += sum_p_dead / na
                            MLE_T[state][action][ns] = MLE_T[state][action][ns]

            for state in MLE_T.keys():
                for action in MLE_T[state].keys():
                    p = np.sum(list(MLE_T[state][action].values()))
                    for ns in MLE_T[state][action].keys():
                        if p != 0:
                            MLE_T[state][action][ns] /= p
                        else:
                            MLE_T[state][action][ns] = 0
            n = dict()
            for episode in batch:
                for (state, action, reward, next_state) in episode:
                    if state not in n:
                        n[state] = dict()
                    if action not in n[state]:
                        n[state][action] = 1
                    else:
                        n[state][action] += 1

            mask = compute_compressed_mask(n, n_wedge)
            # Parameters of mcts
            exp_constant = 5
            gamma = 0.85
            tree_depth = ceil(log(0.01, gamma))
            initial_state = 0
            final_states = [10000000000]
            pib_mcts = Mcts_ext(gamma, env.n_states, env.n_actions, MLE_T, env, mask, 0, n_sim,
                                exploration_costant=exp_constant, max_depth=tree_depth,
                                states_to_sim=[initial_state], final_states=final_states)
            start = time.time()
            pib_mcts.fit()
            end = time.time()

            mcts_times[n_sim].append(end - start)
            print(end - start)

    save_object(mcts_times, 'out/SysAdmin/results/mcts_computational_time.pkl')
    print('MCTS-SPIBB computational time for machines %s' % number_of_machines)
    for k in mcts_times.keys():
        print(mcts_times[k])


if __name__ == '__main__':
    parsed_args = parser.parse_args()
    parsed_args.agents_config = get_configs_from_dicts(parse_json(parsed_args.agents_config))
    parsed_args.behavior_policy_config = parse_json(parsed_args.behavior_policy_config)

    if parsed_args.n_machines >= 13:
        policy_comparison_extended(**vars(parsed_args))
    else:
        policy_comparison(**vars(parsed_args))

    time_comparison(**vars(parsed_args))
