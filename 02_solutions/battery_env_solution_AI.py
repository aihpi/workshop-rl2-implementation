"""
Battery Storage Environment - SOLUTION FILE

This is the complete reference implementation for the workshop.
DO NOT share with participants until after they've attempted the exercise.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BatteryStorageEnv(gym.Env):
    """
    A Gymnasium environment for home battery storage optimization.

    The agent controls a battery system to minimize electricity costs by
    charging when prices are low and discharging when prices are high.

    Episode: configurable length (default 168 steps = 1 week, hourly resolution)

    Action Space:
        Box([-1], [1], float32)
        -1 = maximum discharge, +1 = maximum charge
        Actual power = action * max_charge_rate

    Observation Space:
        Box([0]*obs_dim, [1]*obs_dim, float32)
        [soc_norm, hour_of_day, price_norm, load_norm, forecast_price_1, forecast_load_1, ...]
        All values normalized to [0, 1]

    Reward:
        Negative cost of grid energy: -(grid_energy * price)
        grid_energy = max(0, load + charge_energy)
        Note: Can't sell back to grid, only offset own consumption.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_path: str | Path | None = None,
        capacity: float = 10.0,
        max_charge_rate: float = 2.0,
        efficiency: float = 1.0,
        forecast_horizon: int = 4,
        enable_degradation: bool = False,
        episode_length: int = 168,
        split: str = "train",
        train_fraction: float = 0.80,
    ):
        """
        Initialize the battery storage environment.

        Args:
            data_path: Path to directory containing prices.npy and loads.npy.
                      If None, uses default path relative to this file.
            capacity: Battery capacity in kWh.
            max_charge_rate: Maximum charge/discharge rate in kW.
            efficiency: Round-trip efficiency (1.0 = 100%, no losses).
            forecast_horizon: Number of future steps to include in observation.
            enable_degradation: If True, enables battery degradation (Level 2).
            episode_length: Number of hourly steps per episode (default: 168 = 1 week).
            split: Which data split to use: "train" (default), "eval", or "all".
            train_fraction: Fraction of episodes used for training (default: 0.80).
                           The remaining fraction is used for evaluation.
        """
        super().__init__()

        # Battery parameters
        self.capacity = capacity
        self.max_charge_rate = max_charge_rate
        self.efficiency = efficiency
        self.forecast_horizon = forecast_horizon
        self.enable_degradation = enable_degradation

        # Episode parameters
        self.episode_length = episode_length

        # Load data and chunk into episodes
        self._load_data(data_path)

        # Set available episode range based on split
        split_idx = int(self.n_episodes * train_fraction)
        if split == "train":
            self._available_episodes = (0, split_idx)
        elif split == "eval":
            self._available_episodes = (split_idx, self.n_episodes)
        elif split == "all":
            self._available_episodes = (0, self.n_episodes)
        else:
            raise ValueError(f"split must be 'train', 'eval', or 'all', got '{split}'")

        # Define action space: continuous [-1, 1]
        # -1 = max discharge, +1 = max charge
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Define observation space: all normalized to [0, 1]
        # [soc, hour_of_day, price, load, forecast_price_1, forecast_load_1, ...]
        obs_dim = 4 + 2 * forecast_horizon
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # State variables (initialized in reset)
        self.soc: float = 0.0
        self.current_step: int = 0
        self.episode_idx: int = 0
        self._current_prices: np.ndarray = np.zeros(self.episode_length)
        self._current_loads: np.ndarray = np.zeros(self.episode_length)
        self._current_hours_of_day: np.ndarray = np.zeros(self.episode_length, dtype=np.int64)
        self._current_days_of_week: np.ndarray = np.zeros(self.episode_length, dtype=np.int64)
        self.health: float = 1.0  # Battery health for degradation (Level 2)

    def _load_data(self, data_path: str | Path | None) -> None:
        """
        Load price, load, and hour-of-day data from .npy files and chunk into episodes.

        The raw data is a flat 1D timeseries. This method chunks it into
        episodes of length self.episode_length, discarding any leftover hours
        that don't fill a complete episode.

        Sets:
            self.prices: Shape (n_episodes, episode_length) - chunked price data
            self.loads: Shape (n_episodes, episode_length) - chunked load data
            self.hours_of_day: Shape (n_episodes, episode_length) - hour of day (0-23)
            self.days_of_week: Shape (n_episodes, episode_length) - day of week (0=Mon, 6=Sun)
            self.n_episodes: Number of complete episodes available
            self.price_max: Maximum price for normalization
            self.load_max: Maximum load for normalization
        """
        if data_path is None:
            # Default path: relative to this file
            data_path = Path(__file__).parent.parent / "03_data"
        else:
            data_path = Path(data_path)

        prices_raw = np.load(data_path / "prices.npy").flatten()
        loads_raw = np.load(data_path / "loads.npy").flatten()
        hours_raw = np.load(data_path / "hours_of_day.npy").flatten()
        days_raw = np.load(data_path / "days_of_week.npy").flatten()

        # Chunk into episodes, discarding incomplete trailing hours
        n_usable = (len(prices_raw) // self.episode_length) * self.episode_length
        self.prices = prices_raw[:n_usable].reshape(-1, self.episode_length)
        self.loads = loads_raw[:n_usable].reshape(-1, self.episode_length)
        self.hours_of_day = hours_raw[:n_usable].reshape(-1, self.episode_length)
        self.days_of_week = days_raw[:n_usable].reshape(-1, self.episode_length)
        self.n_episodes = self.prices.shape[0]

        # Compute normalization constants from usable data only
        self.price_max = self.prices.max()
        self.load_max = self.loads.max()

    def _get_info(self) -> dict[str, Any]:
        """
        Get debug information dictionary.

        Returns:
            Dictionary with current state information for debugging.
        """
        # Safe index for terminal state (at episode_length -> use last valid index)
        step_idx = min(self.current_step, self.episode_length - 1)
        return {
            "soc": self.soc,
            "step": self.current_step,
            "episode_idx": self.episode_idx,
            "price": self._current_prices[step_idx],
            "load": self._current_loads[step_idx],
            "hour_of_day": int(self._current_hours_of_day[step_idx]),
            "day_of_week": int(self._current_days_of_week[step_idx]),
            "health": self.health,
            "capacity": self.capacity,
        }

    def _get_forecast(self, horizon: int) -> tuple[float, float]:
        """
        Get (possibly noisy) forecast for a future time step.

        For horizon=0 (current step), returns exact values.
        For horizon>0, adds noise that increases with horizon.

        Args:
            horizon: Steps ahead to forecast (0 = current, 1 = next hour, etc.)

        Returns:
            Tuple of (price_normalized, load_normalized), both in [0, 1]
        """
        # Clamp index to valid range
        future_idx = min(self.current_step + horizon, self.episode_length - 1)

        # Noise increases with horizon: 5% per step ahead
        noise_std = 0.05 * horizon

        # Get normalized values
        price = self._current_prices[future_idx] / self.price_max
        load = self._current_loads[future_idx] / self.load_max

        # Add noise for future forecasts
        if horizon > 0:
            price += self.np_random.normal(0, noise_std)
            load += self.np_random.normal(0, noise_std)

        # Clip to valid range
        return float(np.clip(price, 0, 1)), float(np.clip(load, 0, 1))

    def _apply_degradation(self, charge_power: float) -> None:
        """
        Apply battery degradation based on usage (Level 2 feature).

        This is a stub for the Level 2 workshop extension.
        When enabled, cycling the battery reduces its health over time.

        Args:
            charge_power: The charge/discharge power in kW (can be negative).
        """
        if not self.enable_degradation:
            return

        # Level 2: Implement degradation model
        # Example: health decreases based on energy throughput
        # self.health -= abs(charge_power) * degradation_rate
        pass

    def render(self) -> None:
        """Render the environment (not implemented)."""
        pass

    def close(self) -> None:
        """Clean up resources (not implemented)."""
        pass

    # =========================================================================
    # IMPLEMENTED METHODS (SOLUTION)
    # =========================================================================

    def _get_obs(self) -> np.ndarray:
        """
        Build the observation array for the current state.

        Returns:
            np.ndarray: Observation array of shape (4 + 2*forecast_horizon,)
                       with dtype float32, all values in [0, 1]
        """
        obs = []

        # Current state of charge (normalized)
        soc_norm = self.soc / self.capacity
        obs.append(soc_norm)

        # Safe index for terminal state (at episode_length -> use last valid index)
        step_idx = min(self.current_step, self.episode_length - 1)

        # Hour of day (normalized to [0, 1]) — read from data, not computed from step index
        hour_of_day = self._current_hours_of_day[step_idx] / 24.0
        obs.append(hour_of_day)

        # Current and forecast values
        for h in range(self.forecast_horizon + 1):
            price, load = self._get_forecast(h)
            obs.append(price)
            obs.append(load)

        return np.array(obs, dtype=np.float32)

    def _calculate_reward(
        self, load: float, charge_power: float, price: float
    ) -> float:
        """
        Calculate the reward for the current step.

        Returns:
            float: Negative cost (reward = -cost, so lower cost = higher reward)
        """
        # Grid energy = load + charging (discharge is negative, reduces grid draw)
        grid_energy = load + charge_power

        # Can't sell back to grid - only offset own consumption
        grid_energy = max(grid_energy, 0.0)

        # Cost = energy * price
        cost = grid_energy * price

        # Reward is negative cost (minimize cost -> maximize reward)
        return -cost

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to start a new episode.

        Returns:
            observation: Initial observation from self._get_obs()
            info: Debug info from self._get_info()
        """
        # Initialize RNG (must be first!)
        super().reset(seed=seed)

        # Select random episode from available range
        self.episode_idx = self.np_random.integers(*self._available_episodes)

        # Store episode data
        self._current_prices = self.prices[self.episode_idx].copy()
        self._current_loads = self.loads[self.episode_idx].copy()
        self._current_hours_of_day = self.hours_of_day[self.episode_idx]
        self._current_days_of_week = self.days_of_week[self.episode_idx]

        # Initialize state
        self.soc = self.np_random.uniform(0, self.capacity)
        self.current_step = 0
        self.health = 1.0

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: True if episode ended
            truncated: Always False
            info: Debug info
        """
        # Extract and clip action
        action_value = float(np.clip(action[0], -1.0, 1.0))

        # Convert action to charge power (kW, positive=charging, negative=discharging)
        charge_power = action_value * self.max_charge_rate

        # Calculate new SoC and clip to valid range [0, capacity]
        new_soc = np.clip(self.soc + charge_power, 0, self.capacity)

        # Calculate actual charge power (after constraints)
        actual_charge_power = new_soc - self.soc

        # Update state of charge
        self.soc = new_soc

        # Get current price and load
        price = self._current_prices[self.current_step]
        load = self._current_loads[self.current_step]

        # Calculate reward (using actual charge power after constraints)
        reward = self._calculate_reward(load, actual_charge_power, price)

        # Apply degradation (Level 2)
        self._apply_degradation(actual_charge_power)

        # Advance time
        self.current_step += 1

        # Check termination
        terminated = self.current_step >= self.episode_length
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()
