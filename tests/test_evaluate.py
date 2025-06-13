import pytest
from pathlib import Path
import pandas as pd
from drl_liquidity_sweep.scripts.evaluate import main


def test_evaluate_runs(tmp_path, monkeypatch):
    # Create sample data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    sample_data = pd.DataFrame({
        "time": pd.date_range(start="2024-01-01", periods=100, freq="1S"),
        "bid": 1.0,
        "ask": 1.0002,
        "volume": 1.0
    })
    sample_data.to_csv("data/eurusd_ticks.csv", index=False)
    
    # Create model directory
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # skip heavy plotting in CI
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    
    # Run evaluation
    main("drl_liquidity_sweep/config/default_config.yaml", "models/ppo_liquidity_sweep.zip")
