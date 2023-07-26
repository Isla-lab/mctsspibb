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

import numpy as np
from src import spibb_utils, spibb, modelTransitions
from src.mcts import Mcts
from src.util import load_env_and_policies_gridworld, save_object
from math import ceil, log
import warnings

warnings.filterwarnings("ignore")


def policy_comparison(size):
    n_states = size * size
    n_actions = 4
    nb_trajectories_list = [100, 1000, 10000]
    n_wedge = 20
    n_iterations = 20
    discount_factor = 0.95
    real_T, R, optimal, baseline, garnet, final_states, path = load_env_and_policies_gridworld(size, discount_factor)
    values_baseline = spibb.policy_evaluation_exact(baseline, R, real_T, discount_factor)[0][0]
    values_optimal = spibb.policy_evaluation_exact(optimal.pi, R, real_T, discount_factor)[0][0]
    perf_baseline = [values_baseline] * n_iterations
    perf_optimal = [values_optimal] * n_iterations
    save_object(perf_baseline, 'out/GridWorld/results/v0_baseline_%s.pkl' % size)
    save_object(perf_optimal, 'out/GridWorld/results/v0_optimal_%s.pkl' % size)
    for nb_trajectories in nb_trajectories_list:
        perf_spibb = []
        perf_mcts = dict()
        for it in range(n_iterations):
            trajectories, batch_traj = spibb_utils.generate_batch(nb_trajectories, garnet, baseline)
            model = modelTransitions.ModelTransitions(batch_traj, n_states, n_actions)
            save_object(model.transitions, 'out/GridWorld/mle/gridworld_%s_trj_%s_mle_T_it_%s.pkl'
                        % (size, nb_trajectories, it))
            mask, q = spibb.compute_mask_N_wedge(n_states, n_actions, n_wedge, batch_traj)
            save_object(mask, 'out/GridWorld/mask/gridworld_%s_trj_%s_mask_it_%s.pkl'
                        % (size, nb_trajectories, it))
            save_object(q, 'out/GridWorld/q/gridworld_%s_trj_%s_q_it_%s.pkl'
                        % (size, nb_trajectories, it))
            save_object(model.transitions, 'out/GridWorld/batch/gridworld_%s_trj_%s_batch_it_%s.pkl'
                        % (size, nb_trajectories, it))
            # SPIBB
            seed = 1
            np.random.seed(seed)
            pib_SPIBB = spibb.spibb(discount_factor, n_states, n_actions, baseline, mask, model.transitions, R,
                                    'Pi_b_SPIBB')
            pib_SPIBB.fit()
            values_spibb = spibb.policy_evaluation_exact(pib_SPIBB.pi, R, real_T, discount_factor)[0][0]
            perf_spibb.append(values_spibb)

            # MCTS
            n_sims = [100, 1000, 10000]
            exp_constant = 1
            type_node = 'uniform'
            tree_depth = ceil(log(0.01, discount_factor))
            q_values = np.zeros((n_states, n_actions))
            for n_sim in n_sims:
                perf_mcts[n_sim] = []
                seed = 1
                np.random.seed(seed)
                states_to_sim = [l for l in range(n_states)]

                pib_MCTS = Mcts(discount_factor, n_states,
                                n_actions, baseline, model.transitions, R, mask, q_values, 0, seed,
                                'default', n_sims=n_sim,
                                exploration_costant=exp_constant,
                                max_depth=tree_depth,
                                type_node=type_node,
                                states_to_sim=states_to_sim,
                                final_states=final_states)
                pib_MCTS.fit()

                values_mcts = spibb.policy_evaluation_exact(pib_MCTS.pi, R, real_T, discount_factor)[0][0]
                perf_mcts[n_sim].append(values_spibb - values_mcts)

                print('Value of Optimal policy: %s' % values_optimal)
                print('Value of Baseline policy: %s' % values_baseline)
                print('Value of SPIBB: %s' % values_spibb)
                print('Value of MCTS-SPIBB_n_sims = %s: %s' % (n_sim, values_mcts))
                print('Delta values SPIBB - MCTS-SPIBB_n_sims = %s = %s' % (n_sim, values_spibb - values_mcts))

        save_object(perf_mcts, 'out/GridWorld/results/v0_delta_spibb_mcts_trj_%s_it_%s.pkl' % (nb_trajectories, it))


if __name__ == '__main__':
    gridsize = [3, 4, 5]
    for g in gridsize:
        policy_comparison(g)
