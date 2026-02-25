# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [1.0.0] - 2026-02-25

Initial release of the workshop materials.

### Added
- Battery storage Gymnasium environment (skeleton for participants + reference solution)
- 5 guided Jupyter notebooks (data exploration, heuristic design, environment implementation, RL training, battery degradation)
- 10 years of synthetic hourly electricity price and household load data
- Baseline policies (heuristic, linear programming, model predictive control)
- RL training with PPO/SAC via Stable-Baselines3
- Test suite for environment validation (21 tests)
- Alternative real-world datasets (SMARD, demandlib) for exploration
- Workshop README with problem description, setup instructions, and RL formulation

### Removed
- Docker setup (simplified to uv-based installation)
- Template boilerplate files
