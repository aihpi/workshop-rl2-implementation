"""
Linear Programming (LP) and Model Predictive Control (MPC) baseline policies for the battery scheduling problem (Level 1).

These baselines provide optimal or near-optimal comparison points for the RL
workshop. They solve the battery scheduling problem using classical optimization
(linear programming) rather than reinforcement learning.

Two policies are provided:

1. LP (Linear Programming) with perfect foresight:
   Solves for the entire episode at once, knowing all future prices and loads.
   This gives the globally optimal schedule — no policy can do better.
   Use this as the theoretical upper bound on performance.

2. MPC (Model Predictive Control) with rolling horizon:
   At each timestep, solves a small LP looking only a few steps ahead, then
   executes the first action. More realistic than full LP since it doesn't
   require perfect knowledge of the entire future.

Both policies are factory functions that return callables compatible with the
run_episode() function in the exploration notebook.

Usage:
    from baselines import make_lp_policy, make_mpc_policy

    env = BatteryStorageEnv(episode_length=730)

    lp_policy = make_lp_policy(env)
    data_lp, reward_lp = run_episode(env, policy=lp_policy, seed=42)

    mpc_policy = make_mpc_policy(env, horizon=4)
    data_mpc, reward_mpc = run_episode(env, policy=mpc_policy, seed=42)

Requires: scipy (for scipy.optimize.linprog)
"""

import numpy as np
from scipy.optimize import linprog


def _solve_battery_lp(
    prices: np.ndarray,
    loads: np.ndarray,
    soc_initial: float,
    capacity: float,
    max_charge_rate: float,
) -> np.ndarray:
    """
    Solve the battery scheduling problem as a Linear Program (LP).

    Given a sequence of electricity prices and household loads, find the optimal
    charge/discharge schedule that minimizes total electricity cost from the grid.

    The key idea: we split the battery action into two separate non-negative
    variables (charge_power and discharge_power) to keep all constraints linear.
    The LP solver then finds the combination that minimizes cost.

    Physical setup:
        - Each timestep is 1 hour, so power (kW) equals energy (kWh)
        - The battery can charge or discharge up to max_charge_rate per step
        - Energy balance: grid_energy = load + charge_power - discharge_power
        - Can't sell back to grid: grid_energy >= 0
        - Battery SoC must stay within [0, capacity] at all times

    Decision variables (3 per timestep, 3*T total):
        charge_power[t]    : power used to charge the battery at step t (kW, >= 0)
        discharge_power[t] : power discharged from the battery at step t (kW, >= 0)
        grid_energy[t]     : energy purchased from the grid at step t (kWh, >= 0)

    The decision vector x is laid out as:
        x = [charge_power_0, ..., charge_power_{T-1},       # indices 0..T-1
             discharge_power_0, ..., discharge_power_{T-1},  # indices T..2T-1
             grid_energy_0, ..., grid_energy_{T-1}]          # indices 2T..3T-1

    Args:
        prices: Electricity prices for each timestep, shape (T,), in EUR/kWh.
        loads: Household electricity load for each timestep, shape (T,), in kWh.
        soc_initial: Battery state of charge at the start, in kWh.
        capacity: Maximum battery capacity, in kWh.
        max_charge_rate: Maximum charge/discharge power, in kW.

    Returns:
        Optimal actions for each timestep, shape (T,), in [-1, 1].
        Positive values = charge, negative values = discharge.
        Multiply by max_charge_rate to get actual power in kW.

    Raises:
        RuntimeError: If the LP solver fails to find a feasible solution.
    """
    n_steps = len(prices)

    # =========================================================================
    # OBJECTIVE FUNCTION
    # =========================================================================
    # Minimize total cost: sum of (price[t] * grid_energy[t]) over all steps.
    # Only grid_energy variables appear in the cost — charge_power and
    # discharge_power have zero cost coefficients.
    cost_coefficients = np.zeros(3 * n_steps)
    cost_coefficients[2 * n_steps : 3 * n_steps] = prices  # grid_energy costs

    # =========================================================================
    # VARIABLE BOUNDS
    # =========================================================================
    # charge_power[t]:    0 <= charge_power <= max_charge_rate
    # discharge_power[t]: 0 <= discharge_power <= max_charge_rate
    # grid_energy[t]:     0 <= grid_energy <= infinity (no upper limit)
    variable_bounds = (
        [(0, max_charge_rate)] * n_steps      # charge_power bounds
        + [(0, max_charge_rate)] * n_steps    # discharge_power bounds
        + [(0, None)] * n_steps               # grid_energy bounds (>= 0, no sell-back)
    )

    # =========================================================================
    # INEQUALITY CONSTRAINTS: A_inequality @ x <= b_inequality
    # =========================================================================
    # We have 3 blocks of constraints, each with n_steps rows:
    #   Block 1: Energy balance (can't sell back to grid)
    #   Block 2: SoC upper bound (battery can't exceed capacity)
    #   Block 3: SoC lower bound (battery can't go below zero)
    n_constraints = 3 * n_steps  # T energy balance + T SoC upper + T SoC lower
    n_variables = 3 * n_steps    # T charge + T discharge + T grid_energy
    A_inequality = np.zeros((n_constraints, n_variables))
    b_inequality = np.zeros(n_constraints)

    # --- Block 1: Energy balance ---
    # The grid must supply enough energy for the load plus charging,
    # minus what the battery discharges:
    #
    #   grid_energy[t] >= load[t] + charge_power[t] - discharge_power[t]
    #
    # Rearranged into the LP standard form (A @ x <= b):
    #   charge_power[t] - discharge_power[t] - grid_energy[t] <= -load[t]
    #
    # This also enforces the "can't sell back" constraint: if discharge
    # exceeds load, grid_energy is forced to 0 (its lower bound), and
    # the excess discharge is effectively wasted.
    identity = np.eye(n_steps)
    A_inequality[0:n_steps, 0:n_steps] = identity                        # + charge_power
    A_inequality[0:n_steps, n_steps:2 * n_steps] = -identity             # - discharge_power
    A_inequality[0:n_steps, 2 * n_steps:3 * n_steps] = -identity         # - grid_energy
    b_inequality[0:n_steps] = -loads                                      # <= -load

    # --- Block 2 & 3: State of charge (SoC) bounds ---
    # The SoC at any time t equals the initial SoC plus the cumulative
    # net charge up to that point:
    #
    #   soc[t] = soc_initial + sum(charge_power[0..t]) - sum(discharge_power[0..t])
    #
    # We need: 0 <= soc[t] <= capacity for all t.
    #
    # The cumulative sum is expressed using a lower-triangular matrix:
    #   cumsum(x[0..t]) = L @ x   where L[i,j] = 1 if j <= i, else 0
    #
    # This turns the SoC constraints into linear inequalities.
    cumulative_sum_matrix = np.tril(np.ones((n_steps, n_steps)))

    # Block 2: SoC upper bound — battery can't exceed capacity
    #   soc_initial + cumsum(charge_power) - cumsum(discharge_power) <= capacity
    #   => cumsum(charge_power) - cumsum(discharge_power) <= capacity - soc_initial
    A_inequality[n_steps:2 * n_steps, 0:n_steps] = cumulative_sum_matrix              # + cumsum(charge)
    A_inequality[n_steps:2 * n_steps, n_steps:2 * n_steps] = -cumulative_sum_matrix   # - cumsum(discharge)
    b_inequality[n_steps:2 * n_steps] = capacity - soc_initial

    # Block 3: SoC lower bound — battery can't go below zero
    #   soc_initial + cumsum(charge_power) - cumsum(discharge_power) >= 0
    #   => cumsum(discharge_power) - cumsum(charge_power) <= soc_initial
    A_inequality[2 * n_steps:3 * n_steps, 0:n_steps] = -cumulative_sum_matrix             # - cumsum(charge)
    A_inequality[2 * n_steps:3 * n_steps, n_steps:2 * n_steps] = cumulative_sum_matrix    # + cumsum(discharge)
    b_inequality[2 * n_steps:3 * n_steps] = soc_initial

    # =========================================================================
    # SOLVE THE LP
    # =========================================================================
    result = linprog(
        c=cost_coefficients,
        A_ub=A_inequality,
        b_ub=b_inequality,
        bounds=variable_bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"LP solver failed: {result.message}")

    # =========================================================================
    # EXTRACT ACTIONS
    # =========================================================================
    # The LP gives separate charge and discharge power. Convert to the env's
    # single action format: action = net_power / max_charge_rate
    #   action > 0 means charging, action < 0 means discharging
    #
    # Note: the LP will never have both charge_power > 0 and discharge_power > 0
    # at the same timestep (that would waste energy without reducing cost).
    charge_power = result.x[0:n_steps]
    discharge_power = result.x[n_steps:2 * n_steps]
    net_power = charge_power - discharge_power

    actions = np.clip(net_power / max_charge_rate, -1.0, 1.0)

    return actions


def make_lp_policy(env):
    """
    Create a Linear Programming (LP) policy with perfect foresight.

    This policy solves a single LP over the entire episode, knowing all future
    prices and loads. It then replays the precomputed optimal actions step by
    step. This represents the theoretical best possible performance — no online
    policy can achieve lower cost.

    The LP is solved lazily on the first call after env.reset(). If run_episode()
    is called multiple times with the same policy, the LP automatically re-solves
    for each new episode (detected by env.current_step resetting to 0).

    Args:
        env: A BatteryStorageEnv instance. The policy captures a reference to
             read env._current_prices, env._current_loads, env.soc, etc.

    Returns:
        A callable policy(obs) -> action compatible with run_episode().
    """
    state = {"precomputed_actions": None, "replay_index": 0}

    def policy(obs):
        # Detect new episode: env.current_step == 0 means reset() was just called
        if state["precomputed_actions"] is None or env.current_step == 0:
            state["precomputed_actions"] = _solve_battery_lp(
                prices=env._current_prices,
                loads=env._current_loads,
                soc_initial=env.soc,
                capacity=env.capacity,
                max_charge_rate=env.max_charge_rate,
            )
            state["replay_index"] = 0

        action = state["precomputed_actions"][state["replay_index"]]
        state["replay_index"] += 1
        return np.array([action], dtype=np.float32)

    return policy


def make_mpc_policy(env, horizon=4, use_perfect_forecast=False):
    """
    Create a Model Predictive Control (MPC) policy with rolling horizon.

    At each timestep, this policy solves a small LP looking `horizon` steps
    into the future, then executes only the first action. At the next step,
    it re-solves with updated state (current SoC) and a shifted time window.

    This is more realistic than the full LP because it only uses near-future
    information. With horizon=4, the MPC "sees" the same number of future
    steps as the RL agent's forecast_horizon, making for a fair comparison:
    same information, but MPC has an explicit model of the system dynamics.

    By default, the MPC uses noisy forecasts (same noise the RL agent sees
    via env._get_forecast). Set use_perfect_forecast=True to give MPC exact
    future values instead — useful for isolating the effect of forecast noise
    vs. horizon length.

    Near the end of an episode, the horizon automatically shrinks to fit
    the remaining steps.

    Args:
        env: A BatteryStorageEnv instance. The policy captures a reference to
             read env state (soc, current_step, prices, loads, etc.).
        horizon: Number of steps to look ahead when planning. Default 4.
        use_perfect_forecast: If False (default), uses the same noisy forecasts
            that the RL agent receives via env._get_forecast(). This makes
            the comparison fair: same information quality, but MPC has an
            explicit model of the dynamics.
            If True, uses exact future prices/loads within the horizon window.

    Returns:
        A callable policy(obs) -> action compatible with run_episode().
    """

    def _get_forecast_window(current_step, effective_horizon):
        """Get price and load arrays for the planning window.

        With noisy forecasts (default): uses env._get_forecast() which adds
        noise that increases with horizon (5% std per step ahead), matching
        what the RL agent sees.

        With perfect foresight: reads exact future values directly from
        the episode data arrays.
        """
        if use_perfect_forecast:
            # Perfect foresight: read exact values from the episode data
            window_prices = env._current_prices[current_step : current_step + effective_horizon]
            window_loads = env._current_loads[current_step : current_step + effective_horizon]
            return window_prices, window_loads
        else:
            # Use the same noisy forecast mechanism the RL agent gets.
            # _get_forecast(h) returns (price_normalized, load_normalized)
            # in [0, 1]. We denormalize back to physical units for the LP.
            window_prices = np.zeros(effective_horizon)
            window_loads = np.zeros(effective_horizon)
            for h in range(effective_horizon):
                price_norm, load_norm = env._get_forecast(h)
                window_prices[h] = price_norm * env.price_max
                window_loads[h] = load_norm * env.load_max
            return window_prices, window_loads

    def policy(obs):
        current_step = env.current_step
        remaining_steps = env.episode_length - current_step

        # Shrink horizon if near end of episode
        effective_horizon = min(horizon, remaining_steps)

        if effective_horizon <= 0:
            return np.array([0.0], dtype=np.float32)

        # Get price and load forecasts for the planning window
        window_prices, window_loads = _get_forecast_window(current_step, effective_horizon)

        # Solve LP for this short window
        optimal_actions = _solve_battery_lp(
            prices=window_prices,
            loads=window_loads,
            soc_initial=env.soc,
            capacity=env.capacity,
            max_charge_rate=env.max_charge_rate,
        )

        # Execute only the first action (MPC principle: plan ahead, act now)
        return np.array([optimal_actions[0]], dtype=np.float32)

    return policy
