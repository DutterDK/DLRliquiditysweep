"""
Evaluate a trained PPO liquidity-sweep model on a hold-out slice.

Usage as CLI:
    python -m drl_liquidity_sweep.scripts.evaluate \
        --config config/exp_2025-long.yaml \
        --model  models/ppo_5000000.zip \
        [--vecnorm models/vecnorm.pkl] [--no-plot]

The module also exposes `main()` for backward-compat tests:
    main(cfg_path, model_path, vec) → map them.
    • If only kwargs supplied (new CLI) → pass through.
"""
from pathlib import Path
import yaml, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from drl_liquidity_sweep.data.loader import load_tick_csv
from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv
import json
import inspect

def _evaluate(cfg_path, model_path, vecnorm_path=None, plot=True, out_metrics=None):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    df  = load_tick_csv(cfg["env"]["data_file"], to_seconds=cfg["env"]["bar_seconds"])
    # slice hold-out dates if specified
    if "test_start" in cfg["env"] and "test_end" in cfg["env"]:
        df  = df.loc[cfg["env"]["test_start"]:cfg["env"]["test_end"]]
    env = LiquiditySweepEnv(
        df,
        lambda_dd = cfg["env"]["lambda_dd"],
        commission= cfg["env"]["commission"],
        reset_on_day_change=True,
    )
    if vecnorm_path:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        venv = DummyVecEnv([lambda: env])
        venv = VecNormalize.load(vecnorm_path, venv)
        env  = venv
    model = PPO.load(model_path, env=env)

    # Handle reset for both raw and vectorized envs
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs = reset_result[0]
    else:
        obs = reset_result

    rewards = []
    equity = [0.0]
    positions = []
    trade_profits = []
    trade_durations = []
    trade_open_step = None
    last_position = 0
    done = False
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        is_vec = hasattr(env, 'envs')
        action_in = [action] if is_vec else int(action)
        step_result = env.step(action_in)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            next_obs, r, done_flag, _, _ = step_result
        else:
            next_obs, r, done_flag, _ = step_result
        # Unpack if using VecEnv
        if isinstance(r, (list, np.ndarray)):
            r = r[0]
            done_flag = done_flag[0]
            next_obs = next_obs[0]
        rewards.append(r)
        equity.append(equity[-1] + r)
        positions.append(env.position if not is_vec else env.envs[0].position)
        # Trade analysis: detect trade close
        if last_position != 0 and (positions[-1] == 0):
            # Trade closed
            trade_pnl = equity[-2] - equity[trade_open_step] if trade_open_step is not None else 0.0
            trade_profits.append(trade_pnl)
            trade_durations.append(step - trade_open_step if trade_open_step is not None else 0)
            trade_open_step = None
        if last_position == 0 and (positions[-1] != 0):
            # Trade opened
            trade_open_step = step
        last_position = positions[-1]
        step += 1
        obs = next_obs
        done = done_flag

    # Final stats
    eq = np.array(equity)
    rewards = np.array(rewards)
    trade_profits = np.array(trade_profits)
    trade_durations = np.array(trade_durations)
    max_dd = np.min(eq - np.maximum.accumulate(eq))
    sharpe = eq[-1] / (np.std(rewards) + 1e-9)
    num_trades = len(trade_profits)
    win_rate = np.mean(trade_profits > 0) if num_trades > 0 else 0.0
    avg_trade = np.mean(trade_profits) if num_trades > 0 else 0.0
    avg_duration = np.mean(trade_durations) if num_trades > 0 else 0.0
    largest_win = np.max(trade_profits) if num_trades > 0 else 0.0
    largest_loss = np.min(trade_profits) if num_trades > 0 else 0.0

    print(f"\n--- Trade Analysis ---")
    print(f"Number of trades: {num_trades}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average trade P/L: {avg_trade:.5f}")
    print(f"Average trade duration: {avg_duration:.2f} steps")
    print(f"Max drawdown: {max_dd:.5f}")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Largest win: {largest_win:.5f}")
    print(f"Largest loss: {largest_loss:.5f}")
    print(f"Final equity: {eq[-1]:.5f}")

    if out_metrics:
        metrics = {
            "num_trades": int(num_trades),
            "win_rate": float(win_rate),
            "avg_trade": float(avg_trade),
            "avg_duration": float(avg_duration),
            "max_drawdown": float(max_dd),
            "sharpe": float(sharpe),
            "largest_win": float(largest_win),
            "largest_loss": float(largest_loss),
            "final_equity": float(eq[-1]),
        }
        with open(out_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {out_metrics}")

    if plot:
        plt.plot(eq)
        plt.title("Equity curve – hold-out")
        plt.xlabel("Step"); plt.ylabel("Cumulative P/L")
        plt.tight_layout()
        plt.savefig("equity_holdout.png", dpi=150)
        print("Saved equity_holdout.png")

def main(*args, **kwargs):
    """
    Back-compat:
      • If called with positional args (cfg, model, vec) → map them.
      • If only kwargs supplied (new CLI) → pass through.
    """
    sig = inspect.signature(_evaluate)
    if args:
        # Map positional args to _evaluate signature
        params = list(sig.parameters.keys())
        call_kwargs = {k: v for k, v in zip(params, args)}
        call_kwargs.update(kwargs)
        return _evaluate(**call_kwargs)
    return _evaluate(**kwargs)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model",  required=True)
    p.add_argument("--vecnorm")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--out-metrics")
    cli = p.parse_args()
    main(cfg_path=cli.config,
         model_path=cli.model,
         vecnorm_path=cli.vecnorm,
         plot=not cli.no_plot,
         out_metrics=cli.out_metrics)
