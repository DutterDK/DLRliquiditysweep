def test_day_episode_length():
    import pandas as pd
    from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv

    dt = pd.date_range("2025-01-01", periods=90_000, freq="1s")
    df = pd.DataFrame(
        {"bid": 1, "ask": 1.0002, "mid": 1.0001, "spread": 0.0002, "volume": 1},
        index=dt,
    )
    env = LiquiditySweepEnv(df, lambda_dd=0, commission=0)
    env.reset()
    steps = 0
    done = False
    while not done and steps < 100_000:
        _, _, done, _, _ = env.step(0)
        steps += 1
        print(f"Step: {steps}, Done: {done}")
    if steps >= 100_000:
        print("Safety break hit: done flag never set!")
    assert 80_000 < steps < 90_000  # one full trading day

    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape == (17,)
