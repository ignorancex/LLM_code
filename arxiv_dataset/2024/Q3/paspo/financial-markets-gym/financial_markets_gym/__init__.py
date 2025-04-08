from gym.envs.registration import register

register(
    id="financial-markets-env-v0",
    entry_point="financial_markets_gym.envs:FinancialMarketsEnv",
)
