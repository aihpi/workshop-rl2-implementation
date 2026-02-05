"""
Tests for the Battery Storage Environment.

These tests validate both the skeleton (workshop version) and the solution.
Run with: uv run pytest tests/test_battery_env.py -v
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env


class TestBatteryEnvSolution:
    """Tests for the complete solution environment."""

    @pytest.fixture
    def env(self):
        """Create a solution environment instance."""
        from solutions.battery_env_solution import BatteryStorageEnv

        env = BatteryStorageEnv()
        yield env
        env.close()

    def test_gymnasium_api_compliance(self, env):
        """Test that the environment passes Gymnasium's check_env."""
        # check_env raises exceptions if the env doesn't comply
        check_env(env, skip_render_check=True)

    def test_observation_space_shape(self, env):
        """Test observation space has correct shape."""
        # obs = [soc, hour_of_day, price_0, load_0, price_1, load_1, ...]
        expected_dim = 4 + 2 * env.forecast_horizon
        assert env.observation_space.shape == (expected_dim,)

    def test_action_space_shape(self, env):
        """Test action space has correct shape."""
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0

    def test_reset_returns_valid_observation(self, env):
        """Test that reset returns observation in correct range."""
        obs, info = env.reset(seed=42)

        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_reset_initializes_state(self, env):
        """Test that reset properly initializes state variables."""
        obs, info = env.reset(seed=42)

        assert 0 <= env.soc <= env.capacity
        assert env.current_step == 0
        assert 0 <= env.week_idx < env.n_weeks
        assert env.health == 1.0

    def test_reset_seeding_reproducibility(self, env):
        """Test that same seed produces same initial state."""
        obs1, _ = env.reset(seed=42)
        soc1 = env.soc
        week1 = env.week_idx

        obs2, _ = env.reset(seed=42)
        soc2 = env.soc
        week2 = env.week_idx

        assert soc1 == soc2
        assert week1 == week2
        np.testing.assert_array_equal(obs1, obs2)

    def test_step_returns_correct_types(self, env):
        """Test that step returns correct types."""
        env.reset(seed=42)
        action = np.array([0.5], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_in_bounds(self, env):
        """Test that step returns observation in valid range."""
        env.reset(seed=42)
        action = np.array([0.5], dtype=np.float32)

        obs, _, _, _, _ = env.step(action)

        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_episode_length(self, env):
        """Test that episode terminates after 168 steps."""
        env.reset(seed=42)

        for i in range(167):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert not terminated, f"Episode terminated early at step {i+1}"

        # Step 168 should terminate
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated, "Episode should terminate at step 168"

    def test_charging_increases_soc(self, env):
        """Test that charging action increases state of charge."""
        env.reset(seed=42)
        env.soc = 5.0  # Set to middle value

        initial_soc = env.soc
        action = np.array([1.0], dtype=np.float32)  # Max charge
        env.step(action)

        assert env.soc > initial_soc

    def test_discharging_decreases_soc(self, env):
        """Test that discharging action decreases state of charge."""
        env.reset(seed=42)
        env.soc = 5.0  # Set to middle value

        initial_soc = env.soc
        action = np.array([-1.0], dtype=np.float32)  # Max discharge
        env.step(action)

        assert env.soc < initial_soc

    def test_soc_bounded_by_capacity(self, env):
        """Test that SoC never exceeds capacity."""
        env.reset(seed=42)
        env.soc = env.capacity - 1.0  # Near full

        # Try to overcharge
        for _ in range(10):
            action = np.array([1.0], dtype=np.float32)
            env.step(action)
            assert env.soc <= env.capacity

    def test_soc_bounded_by_zero(self, env):
        """Test that SoC never goes below zero."""
        env.reset(seed=42)
        env.soc = 1.0  # Near empty

        # Try to over-discharge
        for _ in range(10):
            action = np.array([-1.0], dtype=np.float32)
            env.step(action)
            assert env.soc >= 0.0

    def test_reward_is_negative_when_buying(self, env):
        """Test that reward is negative when consuming from grid."""
        env.reset(seed=42)

        # Charging from empty battery means buying from grid
        env.soc = 0.0
        action = np.array([1.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)

        # Reward should be negative (cost is positive)
        assert reward < 0

    def test_reward_cannot_be_positive(self, env):
        """Test that reward can't be positive (can't sell back to grid)."""
        env.reset(seed=42)

        # Full battery, max discharge should not give positive reward
        env.soc = env.capacity
        action = np.array([-1.0], dtype=np.float32)  # Max discharge
        _, reward, _, _, _ = env.step(action)

        # Reward should be <= 0 (can offset load to zero cost, but can't profit)
        assert reward <= 0

    def test_info_dict_contents(self, env):
        """Test that info dict contains expected keys."""
        obs, info = env.reset(seed=42)

        expected_keys = {"soc", "step", "week", "price", "load", "health"}
        assert expected_keys.issubset(info.keys())

    def test_forecast_horizon_affects_observation_size(self):
        """Test that different forecast horizons produce different obs sizes."""
        from solutions.battery_env_solution import BatteryStorageEnv

        env1 = BatteryStorageEnv(forecast_horizon=2)
        env2 = BatteryStorageEnv(forecast_horizon=6)

        # obs = [soc, hour_of_day, price_0, load_0, ...]
        assert env1.observation_space.shape[0] == 4 + 2 * 2  # 8
        assert env2.observation_space.shape[0] == 4 + 2 * 6  # 16

        env1.close()
        env2.close()

    def test_different_weeks_have_different_data(self, env):
        """Test that different weeks produce different price/load data."""
        obs1, _ = env.reset(seed=42)
        prices1 = env._current_prices.copy()

        # Keep resetting until we get a different week
        env.reset(seed=43)
        prices2 = env._current_prices.copy()

        # Very unlikely to be identical (different seeds should give different weeks)
        assert not np.array_equal(prices1, prices2)


class TestBatteryEnvSkeleton:
    """Tests for the skeleton environment (should fail until implemented)."""

    @pytest.fixture
    def skeleton_env(self):
        """Create a skeleton environment instance."""
        from workshop.envs.battery_env import BatteryStorageEnv

        env = BatteryStorageEnv()
        yield env
        env.close()

    def test_skeleton_init_works(self, skeleton_env):
        """Test that skeleton __init__ works without errors."""
        assert skeleton_env.capacity == 10.0
        assert skeleton_env.max_charge_rate == 5.0
        assert skeleton_env.episode_length == 168

    def test_skeleton_spaces_defined(self, skeleton_env):
        """Test that action and observation spaces are defined."""
        assert skeleton_env.action_space is not None
        assert skeleton_env.observation_space is not None

    def test_skeleton_data_loaded(self, skeleton_env):
        """Test that price and load data is loaded."""
        assert skeleton_env.prices.shape[1] == 168
        assert skeleton_env.loads.shape[1] == 168
        assert skeleton_env.n_weeks > 0

    def test_skeleton_get_forecast_works(self, skeleton_env):
        """Test that _get_forecast is implemented."""
        skeleton_env.reset = lambda seed=None, options=None: (None, None)
        skeleton_env.np_random = np.random.default_rng(42)
        skeleton_env.current_step = 0
        skeleton_env._current_prices = skeleton_env.prices[0]
        skeleton_env._current_loads = skeleton_env.loads[0]

        price, load = skeleton_env._get_forecast(0)
        assert 0 <= price <= 1
        assert 0 <= load <= 1

    def test_skeleton_reset_raises_not_implemented(self, skeleton_env):
        """Test that skeleton reset raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            skeleton_env.reset()

    def test_skeleton_step_raises_not_implemented(self, skeleton_env):
        """Test that skeleton step raises NotImplementedError."""
        # Need to bypass reset to test step
        skeleton_env.np_random = np.random.default_rng(42)
        skeleton_env.current_step = 0
        skeleton_env._current_prices = skeleton_env.prices[0]
        skeleton_env._current_loads = skeleton_env.loads[0]

        with pytest.raises(NotImplementedError):
            skeleton_env.step(np.array([0.0]))

    def test_skeleton_get_obs_raises_not_implemented(self, skeleton_env):
        """Test that skeleton _get_obs raises NotImplementedError."""
        skeleton_env.np_random = np.random.default_rng(42)
        skeleton_env.current_step = 0
        skeleton_env._current_prices = skeleton_env.prices[0]
        skeleton_env._current_loads = skeleton_env.loads[0]

        with pytest.raises(NotImplementedError):
            skeleton_env._get_obs()

    def test_skeleton_calculate_reward_raises_not_implemented(self, skeleton_env):
        """Test that skeleton _calculate_reward raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            skeleton_env._calculate_reward(1.0, 0.5, 0.2)
