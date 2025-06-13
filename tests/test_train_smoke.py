from drl_liquidity_sweep.scripts.train import main


def test_training_smoke(tmp_path, monkeypatch):
    # point config to dummy CSV created on the fly
    csv = tmp_path / "ticks.csv"
    csv.write_text(
        "time,bid,ask,volume\n" + ("2025-01-01 00:00:00,1.0000,1.0002,1\n" * 500)
    )
    conf = tmp_path / "cfg.yaml"
    conf.write_text(
        f"env:\n  data_file: {csv}\n  bar_seconds: 1\n  lambda_dd: 0.0\n  commission: 0.00005\n"
        "ppo:\n  n_steps: 16\n  batch_size: 16\n  learning_rate: 0.0003\n"
        "  gamma: 0.99\n  clip_range: 0.2\n  n_epochs: 10\n  gae_lambda: 0.95\n"
        "misc:\n  seed: 0\n  log_dir: ./logs\n  total_timesteps: 64\n"
    )
    main(conf)  # should run without raising
