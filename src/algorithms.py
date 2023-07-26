import logging
import numpy as np
import time

from src.batch import new_episode


def value_iteration(mdp, epsilon=0.000000001, max_iterations=100000000, initial_q=None, update_policy=True):
    initial_time = time.perf_counter()
    logging.info("Begin value iteration")
    if initial_q is None:
        q_values = np.zeros(shape=(len(mdp.states), len(mdp.actions)))
    else:
        q_values = np.copy(initial_q)

    q_prime = np.zeros(shape=(len(mdp.states), len(mdp.actions)), dtype=np.float)
    greedy_policy = mdp.get_greedy_policy(q_values)
    res = epsilon + 1
    i = 0
    while res > epsilon and i < max_iterations:
        if update_policy:
            greedy_policy = mdp.get_greedy_policy(q_values)
        policy_dist = greedy_policy.full_distribution()
        for s in mdp.states:
            for a in mdp.actions:
                next_state_dist = mdp.next_state_dist(s, a)
                if np.count_nonzero(next_state_dist) > 1:
                    exp_future_reward = np.sum(next_state_dist * q_values * policy_dist)
                else:
                    exp_future_reward = 0
                    for (p, ns) in mdp.successors(s, a):
                        exp_future_reward += np.sum(q_values[ns] * policy_dist[ns] * p)
                q_prime[s, a] = mdp.reward_function(s, a) + mdp.discount_factor * exp_future_reward
        res = np.max(np.abs(q_values - q_prime))

        q_values = np.copy(q_prime)
        if not i % 10:
            logging.info("<VI> iteration: %s, residual: %s, time: %s s", i, res, time.perf_counter() - initial_time)
        i += 1
    logging.info("End value iteration")

    return q_values


def value_iteration_with_budget(mdp, timer, budget, epsilon=0.000000001, max_iterations=100000000, initial_q=None,
                                update_policy=True):
    initial_time = time.perf_counter()
    logging.info("Begin value iteration")
    if initial_q is None:
        q_values = np.zeros(shape=(len(mdp.states), len(mdp.actions)))
    else:
        q_values = np.copy(initial_q)

    q_prime = np.zeros(shape=(len(mdp.states), len(mdp.actions)), dtype=np.float)
    greedy_policy = mdp.get_greedy_policy(q_values)
    res = epsilon + 1
    i = 0
    while res > epsilon and i < max_iterations and time.time() < timer + budget:
        if update_policy:
            greedy_policy = mdp.get_greedy_policy(q_values)
        policy_dist = greedy_policy.full_distribution()
        for s in mdp.states:
            if time.time() < timer + budget:
                for a in mdp.actions:
                    next_state_dist = mdp.next_state_dist(s, a)
                    if np.count_nonzero(next_state_dist) > 1:
                        exp_future_reward = np.sum(next_state_dist * q_values * policy_dist)
                    else:
                        exp_future_reward = 0
                        for (p, ns) in mdp.successors(s, a):
                            exp_future_reward += np.sum(q_values[ns] * policy_dist[ns] * p)
                    q_prime[s, a] = mdp.reward_function(s, a) + mdp.discount_factor * exp_future_reward
            else:
                break
        res = np.max(np.abs(q_values - q_prime))

        q_values = np.copy(q_prime)
        if not i % 10:
            logging.info("<VI> iteration: %s, residual: %s, time: %s s", i, res, time.perf_counter() - initial_time)
        i += 1
    logging.info("End value iteration")

    return q_values


def policy_iteration_with_budget(mdp, budget, epsilon=0.000000001, max_iterations=100000000, initial_q=None,
                                 initial_policy=None):
    initial_time = time.perf_counter()
    logging.info("Begin Policy Iteration")
    if initial_q is None:
        q_values = np.zeros(shape=(len(mdp.states), len(mdp.actions)))
    else:
        q_values = np.copy(initial_q)
    if initial_policy is None:
        policy = mdp.get_greedy_policy(q_values)
    else:
        policy = initial_policy

    max_dif = 1
    i = 0
    timer = time.time()
    while max_dif > 0.001 and i < max_iterations and time.time() < timer + budget:

        q_values = value_iteration_with_budget(mdp, timer, budget, epsilon, max_iterations, initial_q=q_values, update_policy=False)

        new_policy = mdp.get_greedy_policy(q_values)
        max_dif = 0
        for s in mdp.states:
            dist_policy = policy.distribution(s)
            dist_new_policy = new_policy.distribution(s)
            max_dif = max(abs(dist_policy - dist_new_policy).max(), max_dif)

        policy = new_policy
        if not i % 2:
            logging.info("<PI> iteration: %s, probdiff: %s, time: %s s", i, max_dif, time.perf_counter() - initial_time)
        i += 1

    return q_values


def policy_iteration(mdp, epsilon=0.000000001, max_iterations=100000000, initial_q=None, initial_policy=None):
    initial_time = time.perf_counter()
    logging.info("Begin Policy Iteration")
    if initial_q is None:
        q_values = np.zeros(shape=(len(mdp.states), len(mdp.actions)))
    else:
        q_values = np.copy(initial_q)
    if initial_policy is None:
        policy = mdp.get_greedy_policy(q_values)
    else:
        policy = initial_policy

    max_dif = 1
    i = 0

    while max_dif > 0.001 and i < max_iterations:

        q_values = value_iteration(mdp, epsilon, max_iterations, initial_q=q_values, update_policy=False)

        new_policy = mdp.get_greedy_policy(q_values)
        max_dif = 0
        for s in mdp.states:
            dist_policy = policy.distribution(s)
            dist_new_policy = new_policy.distribution(s)
            max_dif = max(abs(dist_policy - dist_new_policy).max(), max_dif)

        policy = new_policy
        if not i % 2:
            logging.info("<PI> iteration: %s, probdiff: %s, time: %s s", i, max_dif, time.perf_counter() - initial_time)
        i += 1
    return q_values


def evaluate_policy(env, policy, discount_factor, repetitions=10 ** 3, max_episode_size=200):
    logging.info("Evaluate policy")
    total_reward = 0
    for i in range(repetitions):
        policy.reset()
        episode = new_episode(env, max_episode_size, policy)
        total_reward += sum(r * discount_factor ** t for t, (_, _, r, _) in enumerate(episode))
    result = total_reward / repetitions
    logging.info("Policy estimated value: {}".format(result))
    return result
