# DRL Liquidity Sweep

A Deep Reinforcement Learning environment for liquidity sweep trading strategies. This project implements a custom Gymnasium environment for training RL agents to execute liquidity sweep trading strategies using PPO (Proximal Policy Optimization).

## Overview

The DRL Liquidity Sweep project provides:
- A custom Gymnasium environment for liquidity sweep trading
- PPO implementation with Stable Baselines3
- TensorBoard logging for training metrics
- Configurable parameters via YAML
- Comprehensive test suite

### What is Liquidity Sweep Trading?

Liquidity sweep trading is a strategy that:
- Identifies key price levels where liquidity is concentrated
- Executes trades when these levels are swept
- Manages positions with drawdown penalties
- Adapts to market conditions through reinforcement learning

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/drl-liquidity-sweep.git
cd drl-liquidity-sweep
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Configure your training parameters in `drl_liquidity_sweep/config/default_config.yaml`:
```yaml
env:
  data_file: "data/your_data.csv"
  bar_seconds: 1

ppo:
  learning_rate: 0.0003
  n_steps: 2048
  # ... other PPO parameters

misc:
  total_timesteps: 1000000
  log_dir: "logs"
```

2. Start training:
```bash
python -m drl_liquidity_sweep.scripts.train
```

3. Monitor training progress with TensorBoard:
```bash
tensorboard --logdir=logs
```

### Evaluation

Evaluate a trained model:
```bash
python -m drl_liquidity_sweep.scripts.evaluate --model_path models/ppo_liquidity_sweep
```

## Development

### Project Structure

```
drl_liquidity_sweep/
├── config/         # Configuration files
│   └── default_config.yaml
├── data/          # Data loading utilities
│   └── loader.py
├── env/           # Trading environment
│   └── liquidity_env.py
├── scripts/       # Training and evaluation scripts
│   ├── train.py
│   └── evaluate.py
└── utils/         # Helper functions
    ├── metrics.py
    └── rewards.py
```

### Running Tests

Run the test suite:
```bash
pytest
```

Run specific test files:
```bash
pytest tests/test_env.py
pytest tests/test_env_commission.py
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

Format code:
```bash
black .
isort .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## TensorBoard

To monitor training progress with TensorBoard:

```sh
pip install tensorboard  # if not already installed
tensorboard --logdir=logs
```

This will launch a local server where you can view training metrics in your browser. 