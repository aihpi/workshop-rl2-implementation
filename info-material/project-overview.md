# RL Workshop Part II: Designing and Building RL Environments

## Workshop Overview

### Title
**Reinforcement Learning II: From Real-World Problems to Working RL Agents**

### Description
In this interactive workshop, participants learn to translate a real-world problem into a Reinforcement Learning setup. Using a simple energy storage optimization example, we go through the complete workflow: formulating the problem as a Markov Decision Process (MDP), implementing a custom Gym environment, training agents with Stable Baselines3, and iteratively improving the solution through reward shaping.

### Target Audience
- Basic Python experience required
- Basic RL knowledge recommended (agents, states, actions, rewards, policies)
- No prior Gym/SB3 experience necessary

### Duration
3 hours (180 minutes)

### Format
- Interactive, hands-on coding workshop
- Participants code along using a provided Docker environment
- Work on the same problem together as a group
- Whiteboard/discussion phases for conceptual design

---

## Learning Objectives

By the end of this workshop, participants will be able to:

### 1. RL Thinking
- Assess when RL is suitable for a given problem
- Recognize when simpler approaches (heuristics, optimization) might suffice
- Understand what makes a problem a "true" RL problem (sequential decisions with long-term consequences)

### 2. MDP Formulation
- Translate a real-world problem into state, action, and reward definitions
- Make informed design choices about state representation and action spaces
- Understand the tradeoffs in reward design (sparse vs. dense, shaping)

### 3. Implementation Skills (Transferable)
- Implement a custom Gym environment following the standard `gymnasium.Env` interface
- Validate environments using SB3's `check_env()`
- Train agents using Stable Baselines3
- Evaluate and compare RL agents against baselines

### 4. Iterative Improvement
- Diagnose why an RL agent might fail or underperform
- Apply reward shaping to guide agent learning
- Understand the difference between environment design and reward shaping

---

## The Problem: Battery Energy Storage Optimization

### Story / Framing
> "You have a home battery (like a Tesla Powerwall). Electricity prices vary throughout the day and week — cheap at night and on weekdays, expensive during evening peaks and weekends. Your household has energy demands that must be met. The task: control the battery (charge/discharge) to minimize your electricity bill over a week.
As input, the agent receives the current battery state, electricity price, and household load. Additionally, the agent receives a NOISY forecast of prices and loads for the next N hours (parameterizable, e.g., N=1 for next hour only, N=3 for next 3 hours. The noise increases with longer horizons).
The agent decides how much to charge or discharge the battery each hour. "
"

### Why This Problem?
- Simple enough to understand and implement in 3 hours
- Realistic and relatable (energy costs, sustainability)
- **Intentionally "too easy" for RL** in the simple version — this teaches judgment about when RL is appropriate
- Can be made into a "true RL problem" by adding complexity (health degradation)
- Week-long episodes enable meaningful long-term planning

### Problem Components

| Component | Description |
|-----------|-------------|
| **Battery** | Storage with capacity (kWh), charge/discharge rate limits |
| **Electricity Price** | Time-varying price profile (€/kWh), different on weekdays vs. weekends |
| **Household Load** | Energy demand that must be satisfied each timestep |
| **Grid** | Always available to buy electricity (at current price) |

### Two Complexity Levels

#### Level 1: Simple (No Health Degradation)
- Battery has fixed capacity
- Actions only affect State of Charge (SoC)
- Problem is close to a classical optimization problem
- Simple heuristics perform well
- **Teaching point:** "Is RL necessary here?"

#### Level 2: With Health Degradation
- Aggressive battery use (high charge/discharge rates, deep discharge) degrades battery health
- Health affects effective capacity: `effective_capacity = max_capacity × health`
- Creates true sequential decision problem: short-term savings vs. long-term battery health
- Actions have delayed, accumulating consequences over the week
- This is where RL genuinely adds value over simple heuristics

---

## Technical Specification

As an inspiration, this repository was used: https://github.com/tobirohrer/building-energy-storage-simulation/tree/master.

### Development Setup

**Primary workflow (recommended):**
- Install uv locally (https://docs.astral.sh/uv/)
- Run `uv sync` to install all dependencies
- Work normally in VSCode or your preferred editor

**Fallback (for participants with OS-specific issues):**
- Install Docker
- Run `docker compose up`
- Work inside the container

The project uses uv for Python package management.
The structure follows best practices for gym environments and stable-baselines3 training. A separate folder for participants to work in, and a separate folder with reference solutions is provided.

### Episode Structure

| Aspect | Value | Rationale |
|--------|-------|-----------|
| **Episode length** | 168 steps (7 days) | Captures full weekly cycle |
| **Step duration** | 1 hour | Standard resolution for energy markets |
| **Weekly structure** | Mon-Sun with distinct patterns | Enables multi-day planning |
| **Initial SoC** | 50% of capacity | Neutral starting point |
| **Initial Health** | 100% | Full health at episode start |

### Why 1 Week?

- Captures weekday vs. weekend dynamics
- Agent can learn to prepare for expensive weekends
- Health degradation effects visible over multiple days
- Realistic planning horizon for battery management

### State Space
To Do: 

Something along the lines of:

Fixed components (4 values):
  - soc
  - health
  - hour_of_day (normalized)
  - day_of_week (normalized)

Variable components (depends on forecast_horizon):
  - price[t], price[t+1], ..., price[t+forecast_horizon]   → (1 + forecast_horizon) values
  - load[t], load[t+1], ..., load[t+forecast_horizon]      → (1 + forecast_horizon) values

Total observation size = 4 + 2 × (1 + forecast_horizon)

The forecast should be noisy, with noise increasing for longer horizons.

### Action Space
```python
action_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(1,),
    dtype=np.float32
)

# Interpretation:
# -1.0 = discharge at 100% of max rate (aggressive)
# -0.5 = discharge at 50% of max rate (moderate)
#  0.0 = hold (no charge/discharge)
# +0.5 = charge at 50% of max rate (gentle)
# +1.0 = charge at 100% of max rate (aggressive)
```

**Why continuous?**
- Most realistic (real battery controllers use continuous signals)
- Enables nuanced health mechanic (agent can choose to be gentle)
- Strategic depth: "when" AND "how hard" to charge/discharge
- Transferable skill for participants

### Battery Parameters
```python
max_capacity = 10.0          # kWh (at 100% health)
max_charge_rate = 2.0        # kWh per hour
min_health = 0.5             # Battery never goes below 50% health

# Health degradation (Level 2 only)
aggressive_threshold = 0.7   # Using >70% of max rate is "aggressive"
damage_rate = 0.003          # 0.3% health loss per aggressive action
deep_discharge_threshold = 0.2  # SoC below 20% causes additional damage
```

### Reward Design

**Dense reward (per-step):**
```python
def _calculate_reward(self, electricity_cost, health_damage=0):
    """
    Dense reward signal — agent receives feedback every step.
    
    Level 1: reward = -electricity_cost
    Level 2 (naive): reward = -electricity_cost (ignores health)
    Level 2 (shaped): reward = -electricity_cost - health_weight * health_damage
    """
    reward = -electricity_cost
    
    # Reward shaping (Level 2 only, participants implement)
    # something like:
    # reward = -electicity_cost - health_weight * health_damage
    
    return reward
```

**Why dense, not sparse?**
- Sparse: reward only at episode end (sum of 168 steps)
- Dense: reward every step
- Dense enables faster learning and easier debugging
- **Teaching discussion point:** Credit assignment problem

### Environment Dynamics (Step Function)
```
1. Extract action as charge rate fraction (-1 to +1)

2. Calculate actual charge amount:
   charge_rate = action × max_charge_rate
   Apply constraints:
   - Cannot charge beyond effective capacity
   - Cannot discharge below 0

3. Apply health degradation (Level 2 only):
   IF |action| > aggressive_threshold:
       damage = damage_rate × (|action| - aggressive_threshold) / (1 - aggressive_threshold)
   IF soc < deep_discharge_threshold × effective_capacity:
       damage += damage_rate × 0.5
   health = max(min_health, health - damage)

4. Update battery state:
   soc = soc + charge_rate

5. Calculate grid usage:
   IF charging (charge_rate > 0):
       grid_usage = load + charge_rate  (cover load AND charge battery)
   ELSE (discharging):
       grid_usage = max(0, load + charge_rate)  (battery helps cover load)

6. Calculate cost and reward:
   electricity_cost = grid_usage × price
   reward = -electricity_cost - (health_weight × health_damage if shaping else 0)

7. Advance time:
   current_step += 1
   Update hour_of_day and if necessary day_of_week

8. Check termination:
   truncated = (current_step >= 168)

9. Return observation, reward, terminated=False, truncated, info
```

---

## Data: Price and Load Profiles

We need to synthetically generate realistic electricity price and household load profiles. The data should be stored as long timeseries of hourly values.

### Requirements
- Weekly patterns with weekday/weekend variation
- Multiple weeks to easily generate train/test split
- Clear daily patterns (cheap night, expensive evening)
- Random noise for variety. Noise should be more complex than just a simple gaussian to mimic real-world data.
- No seasonal trends to keep it simple

### Price Profile (€/kWh)
Should be continuous generated with occasional spikes to simulate real market behavior.
```
                    Weekday     Weekend
Night (00-06):      0.08-0.12   0.10-0.15
Morning (07-09):    0.20-0.28   0.25-0.35
Midday (10-16):     0.12-0.18   0.20-0.28
Evening (17-21):    0.28-0.40   0.35-0.50
Night (22-23):      0.10-0.15   0.12-0.18
```

### Load Profile (kWh per hour)
Should be continuous generated with occasional spikes to simulate real market behavior.
```
                    Weekday     Weekend
Night (00-06):      0.2-0.4     0.3-0.5
Morning (07-09):    0.8-1.2     1.0-1.5
Midday (10-16):     0.3-0.5     0.8-1.2    ← People home on weekend
Evening (17-21):    1.2-1.8     1.5-2.2
Night (22-23):      0.4-0.6     0.5-0.8
```

---

## Baselines for Comparison

### 1. Random Agent
```python
action = env.action_space.sample()  # Random value in [-1, 1]
```
Purpose: Lower bound, shows that learning happens.

### 2. Heuristic Agent (Threshold Policy)
```python
def heuristic_action(obs):
    soc, health, price_now, price_next, load_now, load_next, hour, day = obs
    
    # Simple strategy: charge when cheap, discharge when expensive
    price_threshold = 0.5  # Normalized threshold
    
    if price_now < 0.3 and soc < 0.8:
        return 0.7  # Charge moderately
    elif price_now > 0.6 and soc > 0.3:
        return -0.7  # Discharge moderately
    else:
        return 0.0  # Hold
```
Purpose: Simple rule-based approach. Works well in Level 1, struggles with health in Level 2.

### 3. RL Agent (PPO)
```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```
Purpose: The learning approach we teach.

### Expected Results

| Agent | Level 1 (Simple) | Level 2 (With Health) |
|-------|------------------|----------------------|
| Random | Poor | Poor |
| Heuristic | Good | Medium (ignores health) |
| RL (naive reward) | Good | Poor (destroys battery) |
| RL (shaped reward) | Good | Good |

---

## Workshop Flow

### Overview
```
┌─────────────────────────────────────────────────────────────────┐
│ PART 1: Build Simple Environment (Level 1)                      │
│   • MDP formulation on whiteboard                               │
│   • Implement Gym environment                                   │
│   • Train RL agent                                              │
│   • Compare to heuristic baseline                               │
│   • Discussion: "Is this really an RL problem?"                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PART 2: Add Complexity — Health Degradation (Level 2)           │
│   • Enable degradation (environment design)                     │
│   • Train naive RL agent → observe it struggle                  │
│   • Discuss: "Why does it fail?"                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PART 3: Reward Shaping                                          │
│   • Brainstorm reward modifications                             │
│   • Implement reward shaping                                    │
│   • Train improved agent                                        │
│   • Compare all approaches                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Timeline

| Time | Duration | Step | Activities |
|------|----------|------|------------|
| 0:00 | 20 min | **Setup & Introduction** | RL recap, present battery problem, verify Docker works |
| 0:20 | 15 min | **MDP Formulation (Whiteboard)** | Define state, action, reward together as group |
| 0:35 | 45 min | **Implement Environment** | Participants implement `reset()`, `step()`, `_get_obs()`, `_calculate_reward()` |
| 1:20 | 25 min | **Train & Compare** | Train RL agent, run heuristic, compare results, discuss |
| 1:45 | 5 min | **Break** | |
| 1:50 | 15 min | **Add Health Degradation** | Explain concept, enable flag, discuss expected impact |
| 2:05 | 20 min | **Observe Naive RL** | Train on complex env, see failure, discuss why |
| 2:25 | 35 min | **Reward Shaping** | Brainstorm (whiteboard), implement, train, compare |
| 3:00 | 15 min | **Wrap-up** | Results comparison, key takeaways, Q&A |

**Total: ~195 min — adjust by shortening implementation or discussion phases as needed**

---

## Key Discussion Points

### 1. Sparse vs. Dense Rewards (During Step 1: Implement `_calculate_reward`)

**Question to pose:**
> "We're giving the agent a reward every hour. But we actually care about the total weekly cost. Why not just give one reward at the end of the week?"

**Discussion points:**
- Credit assignment: "Which of 168 actions caused the final result?"
- Learning signal: Sparse rewards give zero information for most updates
- Human analogy: "Learning piano with feedback only after a full concert"

**Takeaway:**
> Dense rewards → faster learning, but require domain knowledge to design
> Sparse rewards → true objective, but very slow to learn

### 2. Is This Really an RL Problem? (After Level 1 training)

**Question to pose:**
> "Our RL agent works, but so does the simple heuristic. When is RL actually necessary?"

**Discussion points:**
- Actions must have long-term consequences beyond immediate reward
- Level 1: consequences are immediate and reversible → optimization-like
- When does RL shine? Unknown dynamics, complex state-action interactions, long-term tradeoffs

### 3. Environment Design vs. Reward Shaping (After Level 2 failure)

**Question to pose:**
> "We added health degradation (environment design). Now the agent fails. We'll fix it with reward shaping. What's the difference?"

| Concept | What it is | Example |
|---------|------------|---------|
| **Environment Design** | How we model the world (what happens) | Adding health degradation |
| **Reward Shaping** | How we communicate the goal (what signal) | Adding health penalty to reward |

**Key insight:** Environment design changes the problem. Reward shaping changes how we teach the agent about the same problem.

### 4. Reward Shaping Tradeoffs (During Step 5)

**Questions to discuss:**
- "How much weight should we give to health vs. cost?"
- "What happens if we weight health too much? Too little?"
- "Is there a risk of reward hacking?"

---

## Repository Structure
```
rl-energy-workshop/
│
├── README.md                         # Workshop overview, setup instructions
├── SETUP.md                          # Detailed Docker/setup guide
├── docker-compose.yml
│
├── docker/
│   ├── Dockerfile
│   └── requirements.txt              # gymnasium, stable-baselines3, numpy, matplotlib, etc.
│
├── data/
│   ├── prices.npy                    # Pre-generated price profiles (n_weeks × 168)
│   ├── loads.npy                     # Pre-generated load profiles (n_weeks × 168)
│   └── README.md                     # Data documentation
│
├── workshop/                         # === PARTICIPANTS WORK HERE ===
│   │
│   ├── battery_env.py                # Skeleton: Custom Gym environment
│   │                                 # TODOs: reset(), step(), _get_obs(), _calculate_reward()
│   │
│   ├── train.py                      # Train RL agent with SB3 (mostly complete)
│   │
│   ├── evaluate.py                   # Evaluate and compare agents (pre-built)
│   │
│   ├── baselines.py                  # Heuristic agent implementation (pre-built)
│   │
│   └── utils/
│       ├── visualization.py          # Plotting functions (pre-built)
│       └── data_loader.py            # Load price/load data (pre-built)
│
├── solutions/                        # === REFERENCE SOLUTIONS ===
│   ├── battery_env_simple.py         # Level 1 complete solution
│   ├── battery_env_health.py         # Level 2 complete solution
│   ├── battery_env_shaped.py         # Level 2 + reward shaping
│   └── README.md                     # Explanation of solutions
│
└── tests/
    └── test_env.py                   # Environment validation tests
```

---

## Skeleton Code Structure

THE GOAL IS THAT PARTICIPANTS IMPLEMENT THEIR OWN ENVIRONMENT AND LEARN WHAT IS NEEDED TO DO SO. The following just gives an idea of how a skeleton might look like. (But the current is too specific and participants would not learn much if they just fill in the blanks. They should have to think about what to implement where, and learn the expected gym interface pattern.)

### battery_env.py (What Participants Work On)
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BatteryEnv(gym.Env):
    """
    Battery energy storage environment for home energy optimization.
    
    The agent controls a home battery to minimize electricity costs
    over a one-week episode (168 hourly steps).
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, use_health_degradation=False, use_reward_shaping=False, render_mode=None):
        super().__init__()
        
        # === PARAMETERS (pre-defined) ===
        self.max_capacity = 10.0          # kWh
        self.max_charge_rate = 2.0        # kWh per hour
        self.episode_length = 168         # 1 week = 168 hours
        
        # Health degradation settings
        self.use_health_degradation = use_health_degradation
        self.min_health = 0.5
        self.aggressive_threshold = 0.7
        self.damage_rate = 0.003
        self.deep_discharge_threshold = 0.2
        
        # Reward shaping settings
        self.use_reward_shaping = use_reward_shaping
        self.health_weight = 10.0         # Weight for health penalty in shaped reward
        
        # === SPACES (pre-defined) ===
        # State: [soc, health, price_now, price_next, load_now, load_next, hour, day]
        self.observation_space = spaces.Box(
            low=np.array([0, 0.5, 0, 0, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Continuous action: -1 (full discharge) to +1 (full charge)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # === LOAD DATA (pre-built utility) ===
        self.prices_data, self.loads_data = self._load_data()
        self.n_weeks = len(self.prices_data)
        
        # Normalization constants (for observation normalization)
        self.max_price = 0.50  # €/kWh
        self.max_load = 2.5    # kWh
        
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state for a new episode.
        
        TODO: Implement this method
        1. Call super().reset(seed=seed)
        2. Initialize self.soc to 50% of max_capacity
        3. Initialize self.health to 1.0 (100%)
        4. Initialize self.current_step to 0
        5. Sample a random week index for this episode
        6. Store the week's price and load data
        7. Return initial observation and info dict
        
        Returns:
            observation (np.array): Initial state observation
            info (dict): Auxiliary information
        """
        super().reset(seed=seed)
        
        # === YOUR CODE HERE ===
        
        pass  # Remove this line
        
        # return self._get_obs(), self._get_info()
    
    def step(self, action):
        """
        Execute one time step in the environment.
        
        TODO: Implement this method
        1. Extract charge rate from action (action is array with shape (1,))
        2. Calculate actual charge amount considering:
           - max_charge_rate limit
           - Cannot charge beyond effective capacity
           - Cannot discharge below 0
        3. If health degradation enabled, calculate and apply damage
        4. Update self.soc
        5. Calculate grid usage:
           - If charging: grid covers load AND charges battery
           - If discharging: battery helps cover load
        6. Calculate electricity cost = grid_usage × current_price
        7. Calculate reward using _calculate_reward()
        8. Advance self.current_step
        9. Check if episode is truncated (current_step >= episode_length)
        
        Args:
            action (np.array): Charge rate in [-1, 1], shape (1,)
            
        Returns:
            observation (np.array): New state observation
            reward (float): Reward for this step
            terminated (bool): Whether episode ended (always False for this env)
            truncated (bool): Whether episode was cut off (True after 168 steps)
            info (dict): Auxiliary information
        """
        
        # === YOUR CODE HERE ===
        
        pass  # Remove this line
        
        # return self._get_obs(), reward, False, truncated, self._get_info()
    
    def _get_obs(self):
        """
        Construct observation array from current state.
        
        TODO: Implement this method
        - Return normalized array: [soc, health, price_now, price_next, load_now, load_next, hour, day]
        - All values should be normalized to range [0, 1]
        - Handle edge case for price_next and load_next at end of episode
        
        Returns:
            np.array: Observation vector, shape (8,), dtype float32
        """
        
        # === YOUR CODE HERE ===
        
        pass  # Remove this line
    
    def _calculate_reward(self, electricity_cost, health_damage=0):
        """
        Calculate reward for this step.
        
        TODO: Implement this method
        - Base reward is negative electricity cost (we want to minimize cost)
        - If reward shaping is enabled, also penalize health damage
        
        Args:
            electricity_cost (float): Cost of electricity this step (€)
            health_damage (float): Health damage this step (0 to ~0.003)
            
        Returns:
            float: Reward value
        """
        
        # === YOUR CODE HERE ===
        
        pass  # Remove this line
    
    def _get_info(self):
        """Return auxiliary information (pre-built)."""
        return {
            "soc": self.soc,
            "health": self.health,
            "step": self.current_step,
            "effective_capacity": self.max_capacity * self.health,
        }
    
    def _apply_degradation(self, rate_fraction):
        """
        Apply health degradation based on usage intensity (pre-built).
        
        Args:
            rate_fraction (float): Absolute charge/discharge rate as fraction of max (0 to 1)
            
        Returns:
            float: Health damage this step
        """
        if not self.use_health_degradation:
            return 0.0
        
        damage = 0.0
        
        # Damage from aggressive charging/discharging
        if rate_fraction > self.aggressive_threshold:
            intensity = (rate_fraction - self.aggressive_threshold) / (1 - self.aggressive_threshold)
            damage += self.damage_rate * intensity
        
        # Additional damage from deep discharge
        effective_capacity = self.max_capacity * self.health
        if self.soc < self.deep_discharge_threshold * effective_capacity:
            damage += self.damage_rate * 0.5
        
        # Apply damage
        self.health = max(self.min_health, self.health - damage)
        
        return damage
    
    def _load_data(self):
        """Load price and load data (pre-built)."""
        from utils.data_loader import load_price_load_data
        return load_price_load_data()
```

---

## What Participants Implement vs. Pre-Built

### Participants Implement (Core Learning)

| Component | Why It's Educational |
|-----------|---------------------|
| `reset()` | Understand episode initialization, state setup |
| `step()` core logic | Understand environment dynamics, constraints |
| `_get_obs()` | Think about state representation, normalization |
| `_calculate_reward()` | Core RL concept, later modified for shaping |

### Pre-Built (Time Saving)

| Component | Why Pre-Build |
|-----------|---------------|
| `__init__` with spaces | Boilerplate, show correct pattern |
| `_apply_degradation()` | Complex, not the focus |
| `_get_info()` | Simple helper |
| `_load_data()` | Utility, not RL-related |
| Training loop (`train.py`) | Standard SB3 pattern |
| Evaluation (`evaluate.py`) | Standard pattern |
| Visualization | Time sink |
| Heuristic baseline | For comparison |
| Data generation | Pre-generated |

---

## Dependencies

### Python Packages (requirements.txt)
```
gymnasium>=0.29.0
stable-baselines3>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
tensorboard>=2.13.0
torch>=2.0.0
```

### Docker Setup

The idea is that the entire workshop can be run in a Docker container to avoid setup issues which occur often in workshops with multiple OS and Python versions. The setup shouold be as simple as installing Docker and running `docker-compose up`, and then being able to modify the code on the go and see the results and work inside the container. 

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default command: JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```
```yaml
# docker-compose.yml
services:
  workshop:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8888:8888"
    volumes:
      - ./workshop:/app/workshop
      - ./solutions:/app/solutions
      - ./data:/app/data
```

---

## Training Configuration
```python
# train.py
from stable_baselines3 import PPO

# PPO works well with continuous action spaces
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,        # Steps per update
    batch_size=64,
    n_epochs=10,
    gamma=0.99,          # Discount factor
    verbose=1,
    tensorboard_log="./logs/"
)

# Training budget
# 168 steps/episode × ~600 episodes = 100k steps
model.learn(total_timesteps=100_000)
```

---

## Summary: Key Teaching Points

| Phase | Teaching Point |
|-------|----------------|
| **MDP Formulation** | How to translate a real problem into state/action/reward |
| **Level 1 Implementation** | Standard Gym interface, transferable pattern |
| **Sparse vs. Dense Discussion** | Why we give feedback every step, not just at the end |
| **Level 1 vs. Heuristic** | RL isn't always necessary — judgment matters |
| **Level 2 Failure** | Environment change (health) breaks naive approach |
| **Environment Design vs. Shaping** | Changing the world vs. changing the signal |
| **Reward Shaping** | How to communicate complex goals to the agent |

---

## The Core Message

> "RL is a powerful framework, but it's not always the right tool. Today you learned:
> 1. **The workflow:** How to implement custom Gym environments with SB3 (transferable skill)
> 2. **The judgment:** How to assess whether RL is appropriate for your problem
> 3. **The iteration:** How to improve RL solutions through reward shaping
> 
> The ability to decide when and how to use RL — that's the real skill."