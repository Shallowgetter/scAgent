from ReplenishmentEnv import make_env
env = make_env("sku200.single_store.standard", wrapper_names=["FlattenWrapper"], mode="test")
obs = env.reset()
print(env.observation_space, env.action_space)
print(obs)
