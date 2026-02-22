"""
Tests for the Battery Storage Environment.

These tests validate any BatteryStorageEnv implementation.
The target implementation is controlled via the --solution flag (see conftest.py).

Run with: uv run pytest tests/test_battery_env.py -v
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env


class TestBatteryEnv:
    """Tests for the BatteryStorageEnv implementation."""

    def test_gymnasium_api_compliance(self, env):
        """Test that the environment passes Gymnasium's check_env."""
        check_env(env, skip_render_check=True)

    def test_observation_space_shape(self, env):
        """Test observation space has correct shape."""
        expected_dim = 5 + 2 * env.forecast_horizon
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
        env.reset(seed=42)

        assert 0 <= env.soc <= env.capacity
        assert env.current_step == 0
        assert 0 <= env.episode_idx < env.n_episodes
        assert env.health == 1.0

    def test_reset_seeding_reproducibility(self, env):
        """Test that same seed produces same initial state."""
        obs1, _ = env.reset(seed=42)
        soc1 = env.soc
        episode1 = env.episode_idx

        obs2, _ = env.reset(seed=42)
        soc2 = env.soc
        episode2 = env.episode_idx

        assert soc1 == soc2
        assert episode1 == episode2
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
        """Test that episode terminates after episode_length steps."""
        env.reset(seed=42)

        for i in range(env.episode_length - 1):
            action = env.action_space.sample()
            _, _, terminated, _, _ = env.step(action)
            assert not terminated, f"Episode terminated early at step {i+1}"

        # Final step should terminate
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated, f"Episode should terminate at step {env.episode_length}"

    def test_charging_increases_soc(self, env):
        """Test that charging action increases state of charge."""
        env.reset(seed=42)
        env.soc = 5.0

        initial_soc = env.soc
        env.step(np.array([1.0], dtype=np.float32))

        assert env.soc > initial_soc

    def test_discharging_decreases_soc(self, env):
        """Test that discharging action decreases state of charge."""
        env.reset(seed=42)
        env.soc = 5.0

        initial_soc = env.soc
        env.step(np.array([-1.0], dtype=np.float32))

        assert env.soc < initial_soc

    def test_soc_bounded_by_capacity(self, env):
        """Test that SoC never exceeds capacity."""
        env.reset(seed=42)
        env.soc = env.capacity - 1.0

        for _ in range(10):
            env.step(np.array([1.0], dtype=np.float32))
            assert env.soc <= env.capacity

    def test_soc_bounded_by_zero(self, env):
        """Test that SoC never goes below zero."""
        env.reset(seed=42)
        env.soc = 1.0

        for _ in range(10):
            env.step(np.array([-1.0], dtype=np.float32))
            assert env.soc >= 0.0

    def test_reward_is_negative_when_buying(self, env):
        """Test that reward is negative when consuming from grid."""
        env.reset(seed=42)
        env.soc = 0.0

        _, reward, _, _, _ = env.step(np.array([1.0], dtype=np.float32))

        assert reward < 0

    def test_reward_cannot_be_positive(self, env):
        """Test that reward can't be positive (can't sell back to grid)."""
        env.reset(seed=42)
        env.soc = env.capacity

        _, reward, _, _, _ = env.step(np.array([-1.0], dtype=np.float32))

        assert reward <= 0

    def test_info_dict_contents(self, env):
        """Test that info dict contains expected keys."""
        _, info = env.reset(seed=42)

        expected_keys = {"soc", "step", "episode_idx", "price", "load", "health"}
        assert expected_keys.issubset(info.keys())

    def test_forecast_horizon_affects_observation_size(self, env_cls):
        """Test that different forecast horizons produce different obs sizes."""
        env1 = env_cls(forecast_horizon=2)
        env2 = env_cls(forecast_horizon=6)

        assert env1.observation_space.shape[0] == 5 + 2 * 2  # 8
        assert env2.observation_space.shape[0] == 5 + 2 * 6  # 16

        env1.close()
        env2.close()

    def test_different_episodes_have_different_data(self, env):
        """Test that different episodes produce different price/load data."""
        env.reset(seed=42)
        prices1 = env._current_prices.copy()

        env.reset(seed=43)
        prices2 = env._current_prices.copy()

        assert not np.array_equal(prices1, prices2)

    def test_configurable_episode_length(self, env_cls):
        """Test that episode_length parameter works correctly."""
        for length in [24, 168, 720]:
            env = env_cls(episode_length=length, split="all")
            env.reset(seed=42)
            assert env.episode_length == length

            for _ in range(length - 1):
                _, _, terminated, _, _ = env.step(env.action_space.sample())
                assert not terminated

            _, _, terminated, _, _ = env.step(env.action_space.sample())
            assert terminated
            env.close()

    def test_train_eval_split_no_overlap(self, env_cls):
        """Test that train and eval splits don't overlap."""
        train_env = env_cls(split="train")
        eval_env = env_cls(split="eval")

        train_range = train_env._available_episodes
        eval_range = eval_env._available_episodes

        # Train ends where eval begins
        assert train_range[1] == eval_range[0]
        # Together they cover all episodes
        assert train_range[0] == 0
        assert eval_range[1] == train_env.n_episodes

        train_env.close()
        eval_env.close()

    def test_data_chunked_correctly(self, env):
        """Test that data is chunked into episodes of the right shape."""
        assert env.prices.shape == (env.n_episodes, env.episode_length)
        assert env.loads.shape == (env.n_episodes, env.episode_length)
