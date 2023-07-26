from gym.envs.registration import register

register(
    id='Grid-v0',
    entry_point='envs.grid:GridEnv',
    max_episode_steps=200,
)

register(
    id='TaxiStochastic-v0',
    entry_point='envs.taxi_stochastic:TaxiStochasticEnv',
    max_episode_steps=200,
)

register(
    id='TaxiDangerous-v0',
    entry_point='envs.taxi_dangerous:TaxiDangerousEnv',
    max_episode_steps=200,
)

register(
    id='TaxiExtended-v0',
    entry_point='envs.taxi_extended:TaxiExtendedEnv',
    max_episode_steps=200,
)

register(
    id='TaxiAbsorbing-v0',
    entry_point='envs.taxi_absorbing:TaxiAbsorbingEnv',
    max_episode_steps=200,
)

register(
    id='SysAdmin-v0',
    entry_point='envs.sysadmin:SysAdminEnv',
    max_episode_steps=40,
    kwargs={'size': 8},
)

for i in range(3, 50):
    register(
        id='SysAdmin{}-v0'.format(i),
        entry_point='envs.sysadmin:SysAdminEnv',
        max_episode_steps=40,
        kwargs={'size': i},

    )

for i in range(5):
    for j in range(5):
        register(
            id='StockTrading_{}_{}-v0'.format(i, j),
            entry_point='envs.stock_trading:StockTradingEnv',
            max_episode_steps=40,
            kwargs={'number_of_sectors': i,
                    'number_of_stocks_per_sector': j},

        )
