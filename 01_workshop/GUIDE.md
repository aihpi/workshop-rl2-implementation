# Workshop: Battery Storage Optimization with Reinforcement Learning

## The Problem

You have a home battery (like a Tesla Powerwall). Electricity prices vary throughout the day and week — cheap at night when demand is low, expensive during evening peaks. Your household has energy demands that must be met at all times.

Without a battery, you simply buy electricity from the grid at whatever the current price is. But with a battery, you have a choice: **charge it when electricity is cheap, and discharge it to cover your load when electricity is expensive**. The question is: when exactly should you charge, how much, and when should you discharge?

This is hard to answer with simple rules. Prices and consumption patterns are noisy, partially predictable, and interact in complex ways. This is where reinforcement learning comes in.

## The Goal

Build a **Gymnasium environment** that simulates this problem so that an RL agent can learn an optimal charging strategy through trial and error.

The environment should:

- Simulate one week (168 hours) per episode
- Let the agent decide each hour how much to charge or discharge
- Reward the agent for reducing electricity costs compared to having no battery

Once the environment works, we can train a PPO agent on it and see if it learns to "buy cheap, use when expensive."

## The Data

You have **3 years of hourly data** stored as NumPy arrays:

| File | Shape | Description |
|------|-------|-------------|
| `prices.npy` | (156, 168) | Electricity price per hour (EUR/kWh) |
| `loads.npy` | (156, 168) | Household consumption per hour (kWh) |

- **156 weeks**, each with **168 hours** (7 days x 24 hours)
- Row `i` contains all hourly values for week `i`
- Prices show daily patterns (cheap at night, expensive in evening) and seasonal variation
- Loads follow household routines (low at night, peaks at morning and evening)

The first 130 weeks are used for **training**, the remaining 26 weeks for **evaluation**. This prevents the agent from memorizing specific weeks.

Explore the data in `00_explore_data.ipynb` before building the environment.

## The Grid

The electricity grid is always available. Every hour, the household buys whatever energy it needs from the grid at the current market price. There are no blackouts or supply limits — the grid simply costs money.

The household **cannot sell energy back** to the grid. If the battery discharges more than the household needs, the excess energy is simply wasted. This means the minimum grid purchase in any hour is zero, never negative.

## The Battery

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Capacity | 10 kWh | Maximum energy the battery can store |
| Max charge rate | 5 kW | Maximum power in/out per hour |
| Efficiency | 100% | No energy losses during charge/discharge |
| Initial SoC | Random | Each episode starts with a random charge level |

Since each time step is 1 hour and power is in kW, the energy transferred in one step equals the power value (kWh = kW x 1h).

### Physical constraints

The state of charge (SoC) must always stay between 0 and the battery capacity. This means:

- **Battery full**: If the battery is at 9 kWh and the agent requests +5 kW (full charge), only 1 kWh is actually charged. The environment must enforce this — the *actual* charge power may be less than what the agent requested.
- **Battery empty**: If the battery is at 1 kWh and the agent requests -5 kW (full discharge), only 1 kWh is actually discharged. The household still has to buy the rest of its load from the grid.

The reward should always be based on the **actual** power transferred, not the raw action value.

## The RL Formulation

### What the Agent Observes (State)

Each hour, the agent receives a vector of normalized values (all in [0, 1]):

| Component | Meaning |
|-----------|---------|
| SoC (normalized) | How full is the battery? (0 = empty, 1 = full) |
| Hour of day | What time is it? (0 = midnight, 0.5 = noon) |
| Current price | How expensive is electricity right now? |
| Current load | How much is the household consuming right now? |
| Price forecasts | What might the price be in the next hours? |
| Load forecasts | What might consumption be in the next hours? |

The forecasts get noisier the further ahead they look, just like real weather and price forecasts. The environment provides a helper method `_get_forecast(h)` that returns a noisy (price, load) tuple for `h` steps into the future.

### What the Agent Does (Action)

A single continuous value in **[-1, 1]**:

| Value | Meaning |
|-------|---------|
| -1.0 | Discharge at maximum rate |
| 0.0 | Do nothing |
| +1.0 | Charge at maximum rate |

The actual power is: `charge_power = action * max_charge_rate`

### What the Agent Optimizes (Reward)

Every hour, the household must purchase electricity from the grid to cover:

```
grid_energy = household_load + battery_charging
```

- **Charging** the battery increases grid purchases (you're buying extra electricity to store)
- **Discharging** reduces grid purchases (you're using stored energy to offset your load)
- Grid energy **cannot go below zero** — excess discharge is wasted, not sold

The **cost** this hour is `grid_energy * price`, and the **reward** is the negative cost. The agent maximizes reward, which means it minimizes cost.

Note: The household load is always fully satisfied. The battery doesn't "power the house" — it shifts *when* you buy from the grid. Without a battery, you pay `load * price` every hour. With a battery, you pay `max(0, load + charge_power) * price`.

### Episode Structure

- Each episode is **one week** (168 hourly steps)
- At the start of each episode, a random week is selected from the dataset
- The battery starts with a random charge level
- The episode ends after 168 steps (no early termination)

## Your Task

Open `01_workshop/envs/battery_env.py`. The skeleton provides all the infrastructure: initialization, data loading, forecast generation, and space definitions. **You implement 4 methods** that define the core RL loop.

### Suggested order

Work through the methods in this order. Each one builds on the previous and lets you test incrementally.

---

**1. Observation (`_get_obs`)**

Assemble the observation vector from the current environment state. This is what the agent "sees" each hour. Think about what information a human would want when deciding whether to charge or discharge.

The method should return a NumPy array with all values normalized to [0, 1].

---

**2. Reward (`_calculate_reward`)**

Given the current load, the actual charge power, and the electricity price, compute the reward signal. This defines what "good behavior" means to the agent.

Think about: what is the household's electricity bill this hour? How does the battery affect it?

---

**3. Reset (`reset`)**

Set up a fresh episode. Pick a random week of data, initialize the battery, and return the first observation. This is called at the start of every training episode.

Remember to call `super().reset(seed=seed)` first for proper random number handling.

---

**4. Step (`step`)**

The main loop. Given an action from the agent:
- Convert it to a physical charge/discharge power
- Enforce battery constraints (SoC must stay within bounds)
- Update the battery state
- Compute the reward
- Advance time
- Check if the episode is over

Return the standard Gymnasium 5-tuple: `(observation, reward, terminated, truncated, info)`.

---

## Verification

Once you have all 4 methods implemented, run the test suite:

```bash
uv run pytest tests/test_battery_env.py -v
```

The tests check that your environment:
- Has the correct observation and action spaces
- Returns valid observations within bounds
- Respects battery capacity limits
- Produces sensible rewards (negative, since they represent costs)
- Runs complete episodes of the correct length
- Handles seeding for reproducibility

## What Comes Next

Once the tests pass, open `01_explore_environment.ipynb` to:

1. Compare baseline policies (random, always charge, do nothing, simple heuristic)
2. Train a PPO agent using Stable-Baselines3
3. Visualize the trained agent's behavior and savings
