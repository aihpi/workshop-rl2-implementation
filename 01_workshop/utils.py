from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


# =============================================================================
# Consistent color scheme used across all plots
# =============================================================================

COLOR_PRICE = "tab:blue"
COLOR_LOAD = "black"
COLOR_GRID = "tab:purple"
COLOR_SOC = "tab:green"
COLOR_CHARGE = "tab:green"
COLOR_DISCHARGE = "tab:red"
COLOR_IDLE = "tab:gray"


# =============================================================================
# Episode running and visualization
# =============================================================================


def run_episode(
    env,
    policy: str | Callable = "do_nothing",
    seed: int = 42,
) -> tuple[dict[str, list], float]:
    """Run a single episode and collect data for analysis / plotting.

    Args:
        env: A BatteryStorageEnv instance.
        policy: How to select actions each step. Either:
            - "random": sample from the action space
            - "do_nothing": always take action 0 (no charge/discharge)
            - A callable policy(obs) -> action (np.ndarray of shape (1,))
              This works with any policy factory: make_heuristic_policy,
              make_lp_policy, make_mpc_policy, or a trained PPO model via
              lambda obs: model.predict(obs, deterministic=True)[0]
        seed: Random seed passed to env.reset() for reproducibility.

    Returns:
        data: Dict with lists of per-step data:
            "soc", "price", "load", "health", "capacity",
            "action", "reward", "grid_energy"
        total_reward: Sum of all step rewards (negative = cost).
    """
    obs, info = env.reset(seed=seed)

    data: dict[str, list] = {
        "soc": [info["soc"]],
        "price": [info["price"]],
        "load": [info["load"]],
        "health": [info["health"]],
        "capacity": [info["capacity"]],
        "action": [],
        "reward": [],
        "grid_energy": [],
    }

    total_reward = 0.0
    done = False

    while not done:
        # Select action
        if policy == "random":
            action = env.action_space.sample()
        elif policy == "do_nothing":
            action = np.array([0.0], dtype=np.float32)
        else:
            action = policy(obs)

        # Record pre-step state for grid energy calculation
        old_soc = env.soc
        load = info["load"]

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Calculate actual grid energy from SoC change
        actual_charge_power = env.soc - old_soc
        grid_energy = max(0.0, load + actual_charge_power)

        # Store data
        data["action"].append(float(action[0]))
        data["reward"].append(reward)
        data["grid_energy"].append(grid_energy)

        if not done:
            data["soc"].append(info["soc"])
            data["price"].append(info["price"])
            data["load"].append(info["load"])
            data["health"].append(info["health"])
            data["capacity"].append(info["capacity"])

        total_reward += reward

    return data, total_reward


def plot_episode(data: dict[str, list], title: str = "Episode", xlim=None):
    """Plot episode data collected by run_episode().

    Creates a 5-panel figure showing the full story of a battery episode:
    context (price, load) → decision (actions) → state (SoC) → result (savings).

    Panel order:
        1. Price — electricity price over time
        2. Load / Grid energy — with green/red fill showing battery impact
        3. Actions — charge/discharge/idle bar chart
        4. SoC — battery level with dynamic capacity line
        5. Savings — cumulative savings vs. no-battery baseline

    Args:
        data: Dict returned by run_episode().
        title: Title shown above the top panel.
        xlim: Optional (start, end) tuple to zoom into a time range.

    Returns:
        matplotlib Figure (call plt.show() to display).
    """
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    hours = np.arange(len(data["price"]))
    load = np.array(data["load"])
    grid = np.array(data["grid_energy"])

    # Panel 1: Electricity price
    axes[0].plot(hours, data["price"], COLOR_PRICE, linewidth=1)
    axes[0].set_ylabel("Price (€/kWh)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Load vs. grid energy with fill showing battery impact
    axes[1].plot(hours, load, COLOR_LOAD, linewidth=1, label="Load")
    axes[1].plot(hours, grid, COLOR_GRID, linewidth=1, label="Grid Energy")
    axes[1].fill_between(hours, load, grid, where=grid >= load,
                         alpha=0.2, color=COLOR_CHARGE, label="Charging")
    axes[1].fill_between(hours, load, grid, where=grid < load,
                         alpha=0.2, color=COLOR_DISCHARGE, label="Discharging")
    axes[1].set_ylabel("Energy (kWh)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Agent actions (green=charge, red=discharge)
    colors = [COLOR_CHARGE if a > 0 else COLOR_DISCHARGE if a < 0 else COLOR_IDLE
              for a in data["action"]]
    axes[2].bar(hours, data["action"], color=colors, alpha=0.7, width=1.0)
    axes[2].set_ylabel("Action")
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Battery state of charge with dynamic capacity line
    axes[3].plot(hours, data["soc"], COLOR_SOC, linewidth=1, label="SoC")
    axes[3].plot(hours, data["capacity"], color=COLOR_DISCHARGE, linestyle="--",
                 alpha=0.5, linewidth=1, label="Capacity")
    axes[3].set_ylabel("SoC (kWh)")
    axes[3].set_ylim(0, max(data["capacity"]) * 1.1)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    # Panel 5: Cumulative savings vs. no-battery baseline
    baseline_cost = np.cumsum(np.array(data["price"]) * load)
    actual_cost = np.cumsum(-np.array(data["reward"]))
    savings = baseline_cost - actual_cost
    axes[4].plot(hours, savings, "black", linewidth=1)
    axes[4].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[4].fill_between(hours, 0, savings, where=savings >= 0,
                         alpha=0.2, color=COLOR_CHARGE)
    axes[4].fill_between(hours, 0, savings, where=savings < 0,
                         alpha=0.2, color=COLOR_DISCHARGE)
    axes[4].set_ylabel("Savings (€)")
    axes[4].set_xlabel("Hour")
    axes[4].grid(True, alpha=0.3)

    # Day markers on x-axis
    day_ticks = np.arange(0, len(data["price"]), 24)
    axes[4].set_xticks(day_ticks)
    axes[4].set_xticklabels([f"Day {i // 24 + 1}" for i in day_ticks], rotation=45)

    if xlim:
        for ax in axes:
            ax.set_xlim(xlim)

    plt.tight_layout()
    return fig


# =============================================================================
# Training callback
# =============================================================================


class LiveEvalCallback(BaseCallback):
    """Callback that shows a live cost curve during PPO training.

    Periodically evaluates the current policy on a held-out eval environment
    and updates a matplotlib plot in the Jupyter notebook cell, to visualize the agent's training progress.

    This class inherits from SB3's BaseCallback, which provides hook methods
    that PPO calls at specific points during training. We override two hooks:
    - _on_training_start(): called once before the first training step
    - _on_step(): called after every single env.step() during training

    SB3 BaseCallback docs:
        https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html

    Usage:
        callback = LiveEvalCallback(
            eval_env=eval_env,
            eval_freq=25_000,
            baseline_cost=156.0,
        )
        model.learn(total_timesteps=3_000_000, callback=callback)

    After training, the recorded data is available as:
        callback.timesteps     # list of timestep values where evaluation happened
        callback.mean_rewards  # list of mean rewards at each evaluation point

    Args:
        eval_env: A Gymnasium environment used for evaluation (should use split="eval"
            to avoid evaluating on training data).
        eval_freq: How often to evaluate, in timesteps.
        n_eval_episodes: How many episodes to run per evaluation. More episodes
            give a smoother curve but slow down training. 10 is a good balance.
        baseline_cost: Optional reference cost (e.g., the "no battery" cost) shown
            as a horizontal dashed line on the plot. Helps participants see when
            the agent beats the baseline.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 25_000,
        n_eval_episodes: int = 10,
        baseline_cost: float | None = None,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.baseline_cost = baseline_cost

        # These lists accumulate evaluation results over the course of training.
        # They are plotted as the live curve and can be accessed after training.
        self.timesteps: list[int] = []
        self.mean_rewards: list[float] = []

    def _on_training_start(self) -> None:
        """Called once by PPO before the first training step.

        We use this to evaluate the randomly initialized policy at step 0,
        so the plot shows the starting performance before any learning.
        """
        # evaluate_policy() plays n complete episodes using the current policy
        # and returns (mean_reward, std_reward). We only need the mean.
        mean_r, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,  # no exploration noise, use the agent's best action
        )
        self.timesteps.append(0)
        self.mean_rewards.append(mean_r)
        self._update_plot()

    def _on_step(self) -> bool:
        """Called by PPO after every single env.step() during training.

        With n_envs=4, this fires after every 4 transitions (all envs step
        simultaneously). We check whether we've crossed an eval_freq boundary
        and only run the expensive evaluation then — otherwise we return
        immediately.

        Returns:
            True to continue training. Returning False would abort training.
        """
        # The modulo check with < num_envs handles the case where num_timesteps
        # jumps by num_envs each step (e.g., 4 envs: 4, 8, 12, ...) and might
        # skip the exact multiple of eval_freq.
        if self.num_timesteps % self.eval_freq < self.training_env.num_envs:
            mean_r, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(mean_r)
            self._update_plot()
        return True

    def _update_plot(self):
        """Redraw the live training plot in the current Jupyter cell.

        Uses IPython.display.clear_output(wait=True) to replace the previous
        plot without flickering. The wait=True flag tells Jupyter to keep the
        old output visible until the new output is ready.

        The import is done inside this method (not at module level) so that
        the utils module can be imported in non-Jupyter contexts without error.
        """
        from IPython.display import clear_output, display

        # Clear the previous plot from the cell output
        clear_output(wait=True)

        fig, ax = plt.subplots(figsize=(10, 4))

        # Convert rewards (negative, since cost = -reward) to positive costs
        costs = [-r for r in self.mean_rewards]
        ax.plot(self.timesteps, costs, COLOR_PRICE, linewidth=1.5)

        # Draw the "no battery" baseline as a reference line
        if self.baseline_cost is not None:
            ax.axhline(
                y=self.baseline_cost,
                color=COLOR_DISCHARGE,
                linestyle="--",
                alpha=0.7,
                label=f"No battery: {self.baseline_cost:.0f} €",
            )
            ax.legend(loc="upper right")

        # self.locals is a dict of local variables from PPO's learn() method,
        # provided by BaseCallback. We use it to get total_timesteps for the title.
        total = self.locals.get("total_timesteps", self.num_timesteps)
        current_cost = costs[-1]
        ax.set_title(
            f"Training: {self.num_timesteps / 1e6:.1f}M / {total / 1e6:.1f}M steps "
            f"— Avg cost: {current_cost:.2f} €"
        )
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Avg Cost per Episode (€)")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        # Close the figure to free memory (otherwise matplotlib accumulates them)
        plt.close(fig)
