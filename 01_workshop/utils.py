import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


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
        ax.plot(self.timesteps, costs, "b-", linewidth=1.5)

        # Draw the "no battery" baseline as a reference line
        if self.baseline_cost is not None:
            ax.axhline(
                y=self.baseline_cost,
                color="red",
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
