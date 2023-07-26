import unittest

from envs.grid import GridEnv


class TestGrid(unittest.TestCase):
    def test_initial_state(self):
        grid = GridEnv()
        state = grid.reset()
        self.assertEqual((0, 0), GridEnv.decode(state))

    def test_number_of_actions(self):
        grid = GridEnv()
        number_of_actions = grid.nA
        self.assertEqual(4, number_of_actions)

    def test_actions(self):
        grid = GridEnv()
        state, reward, done, _ = grid.step(grid.action_space.sample())
        self.assertIn(GridEnv.decode(state), [(0, 0), (0, 1), (1, 0)])

    def test_actions_reward(self):
        grid = GridEnv()
        next_state = grid.reset()
        for i in range(200):
            state = next_state
            action = grid.action_space.sample()
            next_state, reward, done, _ = grid.step(action)

            if next_state == state:
                self.assertEqual(-10, reward)
            elif next_state == grid.goal_state:
                self.assertEqual(100, reward)
                self.assertTrue(done)
                next_state = grid.reset()
            else:
                self.assertEqual(0, reward)

    def test_actions_suc_rate(self):
        grid = GridEnv()
        repetitions = 1000
        count_suc = 0
        count_fail = 0
        for _ in range(repetitions):
            grid.reset()
            next_state, _, _, _ = grid.step(0)
            if GridEnv.decode(next_state) == (0, 1):
                count_suc += 1
            else:
                count_fail += 1
        suc_rate = count_suc / repetitions
        fail_rate = count_fail / repetitions
        self.assertAlmostEqual(suc_rate, 0.75, places=1)
        self.assertAlmostEqual(fail_rate, 0.25, places=1)

    def test_actions_effects(self):
        grid = GridEnv(initial_states=[(1, 1)])

        repetitions = 1000
        count_suc = 0
        count_fail_opposite = 0
        count_fail_side1 = 0
        count_fail_side2 = 0
        for _ in range(repetitions):
            s = grid.reset()
            self.assertEqual((1, 1), grid.decode(s))

            next_state, reward, done, _ = grid.step(0)
            if GridEnv.decode(next_state) == (1, 2):
                count_suc += 1
            elif GridEnv.decode(next_state) == (1, 0):
                count_fail_opposite += 1
            elif GridEnv.decode(next_state) == (0, 1):
                count_fail_side1 += 1
            elif GridEnv.decode(next_state) == (2, 1):
                count_fail_side2 += 1
            else:
                raise Exception()
        suc_rate = count_suc / repetitions
        fail_opposite_rate = count_fail_opposite / repetitions
        fail_side1_rate = count_fail_side1 / repetitions
        fail_side2_rate = count_fail_side2 / repetitions
        self.assertAlmostEqual(suc_rate, 0.75, places=1)
        self.assertAlmostEqual(fail_opposite_rate, 0.05, places=1)
        self.assertAlmostEqual(fail_side1_rate, 0.1, places=1)
        self.assertAlmostEqual(fail_side2_rate, 0.1, places=1)

    def test_encode_decode_states(self):
        import itertools
        states = itertools.product(range(5), range(5))

        for x,y in states:
            state_id = GridEnv.encode(x, y)
            new_x, new_y = GridEnv.decode(state_id)
            self.assertEqual(new_x, x)
            self.assertEqual(new_y, y)

        for state_id in range(25):
            (x,y) = GridEnv.decode(state_id)
            encoded_state = GridEnv.encode(x,y)
            self.assertEqual(encoded_state, state_id)




if __name__ == '__main__':
    unittest.main()
