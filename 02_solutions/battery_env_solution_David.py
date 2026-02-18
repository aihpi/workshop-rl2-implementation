"""
Battery Storage Environment for Reinforcement Learning Workshop.

This environment simulates a home battery storage system that must balance
household electricity load with grid purchases while responding to dynamic pricing.

Participants will implement the core RL interface methods:
- _get_obs(): Build the observation array
- _calculate_reward(): Design the reward signal
- reset(): Initialize episodes
- step(): Process actions and update state
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
        max_charge_rate: float = 5.0,
        efficiency: float = 1.0,
        forecast_horizon: int = 4,
        enable_degradation: bool = False,
        episode_length: int = 168,
        split: str = "train",
        train_fraction: float = 0.83,
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
            train_fraction: Fraction of episodes used for training (default: 0.83).
                           The remaining fraction is used for evaluation.
        """
        super().__init__()

        # Battery parameters
        self.capacity = capacity
        self.max_charge_rate = max_charge_rate
        self.efficiency = efficiency
        # make sure forecast_horizon is non-negative
        if forecast_horizon < 0:
            raise ValueError("forecast_horizon must be non-negative integer")
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
        self.health: float = 1.0  # Battery health for degradation (Level 2)

    def _load_data(self, data_path: str | Path | None) -> None:
        """
        Load price and load data from .npy files and chunk into episodes.

        The raw data is a flat 1D timeseries. This method chunks it into
        episodes of length self.episode_length, discarding any leftover hours
        that don't fill a complete episode.

        Sets:
            self.prices: Shape (n_episodes, episode_length) - chunked price data
            self.loads: Shape (n_episodes, episode_length) - chunked load data
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

        # Chunk into episodes, discarding incomplete trailing hours
        n_usable = (len(prices_raw) // self.episode_length) * self.episode_length
        self.prices = prices_raw[:n_usable].reshape(-1, self.episode_length)
        self.loads = loads_raw[:n_usable].reshape(-1, self.episode_length)
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
        idx = min(self.current_step, self.episode_length - 1)
        return {
            "soc": self.soc,
            "step": self.current_step,
            "episode_idx": self.episode_idx,
            "price": self._current_prices[idx],
            "load": self._current_loads[idx],
            "health": self.health,
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

        # Get normalized values
        price = self._current_prices[future_idx] / self.price_max
        load = self._current_loads[future_idx] / self.load_max

        # Noise increases with horizon: 5% per step ahead
        noise_std = 0.05 * horizon

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

    # =========================================================================
    # METHODS FOR PARTICIPANTS TO IMPLEMENT
    # =========================================================================
    def _get_obs(self) -> np.ndarray:
        """
        Build the observation array for the current state.

        The observation should contain:
        1. Normalized state of charge: soc / capacity
        2. Hour of day: (current_step % 24) / 24.0
        3. Normalized current price: from _get_forecast(0)
        4. Normalized current load: from _get_forecast(0)
        5. Forecast values: for h in 1..forecast_horizon:
           - forecast_price_h from _get_forecast(h)
           - forecast_load_h from _get_forecast(h)

        Returns:
            np.ndarray: Observation array of shape (4 + 2*forecast_horizon,)
                       with dtype float32, all values in [0, 1]

        Hints:
            - Use self._get_forecast(h) to get (price, load) tuple for step h
            - self.soc is current state of charge in kWh
            - self.capacity is maximum capacity in kWh
            - self.current_step % 24 gives hour of day (0-23)
            - self.forecast_horizon tells you how many future steps to include
            - Return type must be np.float32 for Gymnasium compatibility

        Example structure for forecast_horizon=2:
            [soc_norm, hour_norm, price_0, load_0, price_1, load_1, price_2, load_2]
        """
        # Normalize state of charge and hour of day
        soc_norm = self.soc / self.capacity
        hour_norm = (self.current_step % 24) / 24.0
        obs = [soc_norm, hour_norm]

        # add current price and load (horizon=0) and future forecasts
        for h in range(self.forecast_horizon + 1):
            price_h, load_h = self._get_forecast(h)
            obs.extend([price_h, load_h])

        return np.array(obs, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to start a new episode.

        Steps to implement:
        1. Call super().reset(seed=seed) to handle seeding properly
        2. Select a random episode index from the available range
        3. Store the episode's price/load data in self._current_prices/loads
        4. Initialize self.soc to a random value in [0, capacity]
        5. Reset self.current_step to 0
        6. Reset self.health to 1.0 (for degradation feature)

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            observation: Initial observation from self._get_obs()
            info: Debug info from self._get_info()

        Hints:
            - Use self.np_random for all random number generation
              (this is set by super().reset(seed=seed))
            - self.np_random.integers(low, high) gives random int in [low, high)
            - self.np_random.uniform(low, high) gives random float
            - self.prices has shape (n_episodes, episode_length)
            - self._available_episodes is a (start, end) tuple of episode indices
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # select a random episode index from available range
        self.episode_idx = self.np_random.integers(*self._available_episodes)

        # load the prices of the specific episode
        self._current_prices = self.prices[self.episode_idx]
        self._current_loads = self.loads[self.episode_idx]

        # initialize soc to random value
        self.soc = self.np_random.uniform(0, self.capacity)

        # reset current step to 0
        self.current_step = 0

        # reset health
        self.health = 1.0

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Steps to implement:
        1. Convert action to charge power: power = action * max_charge_rate
        2. Get current price and load from episode data
        3. Apply battery constraints:
           - Can't charge above capacity
           - Can't discharge below 0
           - Adjust charge_power if needed
        4. Update state of charge: soc += charge_power * efficiency
           (For discharge, efficiency loss: soc += charge_power / efficiency)
        5. Calculate reward using self._calculate_reward()
        6. Optionally apply degradation via self._apply_degradation()
        7. Increment current_step
        8. Check if episode is done (current_step >= episode_length)

        Args:
            action: Action array of shape (1,) with value in [-1, 1]

        Returns:
            observation: Next observation from self._get_obs()
            reward: Reward for this step from self._calculate_reward()
            terminated: True if episode ended naturally (reached episode_length steps)
            truncated: Always False (we don't truncate episodes)
            info: Debug info from self._get_info()

        Hints:
            - action[0] or float(action[0]) extracts the scalar value
            - np.clip(value, low, high) constrains a value to a range
            - For charging: new_soc = soc + power * efficiency
            - Simpler: if efficiency=1.0, just do soc += power
            - self._current_prices[self.current_step] gives current price
            - self._current_loads[self.current_step] gives current load
            - terminated = (self.current_step >= self.episode_length)
        """
        action = action[0]  # extract scalar value of action array
        charge_power = action * self.max_charge_rate
        new_soc = np.clip(self.soc + charge_power, 0, self.capacity)

        # current price and load
        current_price = self._current_prices[self.current_step]
        current_load = self._current_loads[self.current_step]

        # calculate effective power
        charge_power_effective = new_soc - self.soc

        reward = self._calculate_reward(current_load, charge_power_effective, current_price)

        self._apply_degradation(charge_power_effective)

        # increment current step and update soc
        self.soc = new_soc
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _calculate_reward(
        self, load: float, charge_power: float, price: float
    ) -> float:
        """
        Calculate the reward for the current step.

        The reward should reflect the cost of electricity from the grid.
        Grid energy = load + charge_power (charge_power > 0 means charging)

        When charging: we buy extra electricity -> higher cost
        When discharging: we offset load -> lower cost

        Note: We can't sell back to the grid! If discharge exceeds load,
        the excess is wasted. Clamp grid_energy to minimum 0.

        Args:
            load: Current household load in kWh (energy consumed this hour).
            charge_power: Battery charge power in kW (positive=charging).
                         Since step is 1 hour, this equals energy in kWh.
            price: Current electricity price in currency/kWh.

        Returns:
            float: Negative cost (reward = -cost, so lower cost = higher reward)

        Hints:
            - grid_energy = load + charge_power
            - grid_energy = max(grid_energy, 0)  # Can't sell to grid!
            - cost = grid_energy * price
            - reward should be negative (we want to minimize cost)
        """
        grid_energy = max(load + charge_power, 0)  # Can't sell to grid
        cost = grid_energy * price
        return -cost

    def render(self) -> None:
        """Render the environment (not implemented)."""
        pass

    def close(self) -> None:
        """Clean up resources (not implemented)."""
        pass
