import unittest

from envs.stock_trading import StockTradingEnv


class TestStockExchange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = StockTradingEnv(number_of_sectors=3, number_of_stocks_per_sector=2)

    def test_env_initial_state(self):
        self.env.reset()
        self.assertEqual(self.env.nS, 512)
        self.assertEqual(self.env.s, 0)

    def test_encode_decode_state(self):
        initial_state = self.env.reset()
        self.assertEqual(list(self.env.decode(initial_state)), [0] * 9)
        self.assertEqual(list(self.env.decode(511)), [1] * 9)

    def test_reward_without_owning_any_stocks(self):
        self.env.reset()
        state, reward, done, _ = self.env.step(0)
        self.assertFalse(done)
        self.assertEqual(reward, 0)

    def test_reward_owning_one_stock(self):
        self.env.reset()
        state, _, _, _ = self.env.step(0)  # buy first stock
        state_factors = list(self.env.decode(state))
        _, reward, _, _ = self.env.step(6)  # do nothing
        stock1rising, stock2rising = state_factors[1:3]
        self.assertTrue(state_factors[0])
        if stock1rising and stock2rising:
            self.assertEqual(2, reward)
        elif stock2rising or stock1rising:
            self.assertEqual(0, reward)
        else:
            self.assertEqual(-2, reward)

    def test_buy_all_stock(self):
        state = self.env.reset()
        for i in range(3):
            state, _, _, _ = self.env.step(i)

        state_factors = list(self.env.decode(state))
        for sector in range(3):
            own_stock = sector * 3
            self.assertTrue(state_factors[own_stock], msg="sector {} is not owned".format(sector))

    def test_selling_stocks(self):
        state = self.env.reset()
        state_factors = list(self.env.decode(state))
        self.assertFalse(state_factors[0], msg="initially it should not  own the first stock")

        state, _, _, _ = self.env.step(0)
        state_factors = list(self.env.decode(state))
        self.assertTrue(state_factors[0], msg="it should own the first stock after buying it")

        state, _, _, _ = self.env.step(3)
        state_factors = list(self.env.decode(state))
        self.assertFalse(state_factors[0], msg="it should not own the first stock after selling it")


class TestStockExchangeSingleSector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = StockTradingEnv(number_of_sectors=1, number_of_stocks_per_sector=2)
        cls.parents = cls.env.get_true_parents()

    def test_transition_buy_sector(self):
        transition_function = self.env.P[0][0]
        for p, ns, _, _ in transition_function:
            if ns == self.env.encode(1, 0, 0):
                self.assertAlmostEqual(p, 0.81)
            elif (ns == self.env.encode(1, 1, 0)) or (ns == self.env.encode(1, 0, 1)):
                self.assertAlmostEqual(p, 0.09)
            elif ns == self.env.encode(1, 1, 1):
                self.assertAlmostEqual(p, 0.01)
            else:
                self.assertTrue(False, msg="invalid next state")

    def test_transition_do_nothing_or_sell(self):
        for action in [1, 2]:
            transition_function = self.env.P[0][action]
            for p, ns, _, _ in transition_function:
                if ns == self.env.encode(0, 0, 0):
                    self.assertAlmostEqual(p, 0.81)
                elif (ns == self.env.encode(0, 1, 0)) or (ns == self.env.encode(0, 0, 1)):
                    self.assertAlmostEqual(p, 0.09)
                elif ns == self.env.encode(0, 1, 1):
                    self.assertAlmostEqual(p, 0.01)
                else:
                    self.assertTrue(False, msg="invalid next state")

    def test_parents_own_sector(self):
        own_sector = 0
        action_buy_sector, action_sell_sector, action_do_nothing = range(3)
        self.assertEqual(self.parents[(own_sector, action_buy_sector)], tuple())
        self.assertEqual(self.parents[(own_sector, action_sell_sector)], tuple())
        self.assertEqual(self.parents[(own_sector, action_do_nothing)], tuple([own_sector]))

    def test_parents_sector_rising(self):
        for action in range(3):
            self.assertEqual(self.parents[(1, action)], tuple(range(1, 3)))
            self.assertEqual(self.parents[(2, action)], tuple(range(1, 3)))


if __name__ == '__main__':
    unittest.main()
