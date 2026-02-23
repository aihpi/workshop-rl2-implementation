"""
Generate synthetic electricity price and household load profiles.

Generates a continuous flat timeseries with:
- Weekly patterns (weekday/weekend variation)
- Daily patterns (cheap night, expensive evening)
- Autocorrelated noise (smooth hour-to-hour transitions)
- Occasional spikes

The output is a flat 1D array of hourly values. Episode chunking is handled
downstream by the Gymnasium environment based on its episode_length parameter.
"""

import numpy as np
import matplotlib.pyplot as plt


def _get_base_value(hour_of_day: int, is_weekend: bool, weekday_bases: list, weekend_bases: list) -> float:
    """Get base value for a given hour and day type."""
    bases = weekend_bases if is_weekend else weekday_bases

    if hour_of_day < 7:
        return bases[0]  # night early
    elif hour_of_day < 10:
        return bases[1]  # morning
    elif hour_of_day < 17:
        return bases[2]  # midday
    elif hour_of_day < 22:
        return bases[3]  # evening
    else:
        return bases[4]  # night late


def _generate_autocorrelated_noise(n_hours: int, scale: float, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate autocorrelated noise using AR(1) process.

    noise[t] = alpha * noise[t-1] + sqrt(1 - alpha^2) * epsilon

    The sqrt(1 - alpha^2) factor ensures the variance stays at scale^2.
    """
    noise = np.zeros(n_hours)
    noise[0] = rng.normal(0, scale)

    innovation_scale = scale * np.sqrt(1 - alpha**2)
    for t in range(1, n_hours):
        noise[t] = alpha * noise[t-1] + rng.normal(0, innovation_scale)

    return noise


def generate_price_profile(n_hours: int = 87600, seed: int = 42, autocorr: float = 0.7) -> np.ndarray:
    """
    Generate electricity price profile (EUR/kWh) as continuous timeseries.

    Price ranges:
                        Weekday     Weekend
        Night (00-06):  0.08-0.12   0.10-0.15
        Morning (07-09): 0.20-0.28   0.25-0.35
        Midday (10-16): 0.12-0.18   0.20-0.28
        Evening (17-21): 0.28-0.40   0.35-0.50
        Night (22-23):  0.10-0.15   0.12-0.18

    Args:
        n_hours: Number of hours to generate (default: 87600 = 10 years).
        seed: Random seed for reproducibility.
        autocorr: Autocorrelation coefficient (0-1). Higher = smoother transitions.

    Returns:
        np.ndarray: Shape (n_hours,) - price for each hour
    """
    rng = np.random.default_rng(seed)

    # Base prices: [night_early, morning, midday, evening, night_late]
    weekday_base = [0.10, 0.24, 0.15, 0.34, 0.125]
    weekend_base = [0.125, 0.30, 0.24, 0.425, 0.15]

    # Generate continuous timeseries
    prices = np.zeros(n_hours)

    # Generate autocorrelated noise
    noise = _generate_autocorrelated_noise(n_hours, scale=0.03, alpha=autocorr, rng=rng)

    for t in range(n_hours):
        hour_of_day = t % 24
        day_of_week = (t // 24) % 7
        is_weekend = day_of_week >= 5

        base = _get_base_value(hour_of_day, is_weekend, weekday_base, weekend_base)
        prices[t] = base + noise[t]

        # Occasional price spikes (3% chance, reduced since we have smoother noise now)
        if rng.random() < 0.03:
            prices[t] += rng.uniform(0.05, 0.15)

    # Ensure minimum price
    prices = np.maximum(prices, 0.05)

    return prices


def generate_load_profile(n_hours: int = 87600, seed: int = 43, autocorr: float = 0.8) -> np.ndarray:
    """
    Generate household load profile (kWh per hour) as continuous timeseries.

    Load ranges:
                        Weekday     Weekend
        Night (00-06):  0.2-0.4     0.3-0.5
        Morning (07-09): 0.8-1.2     1.0-1.5
        Midday (10-16): 0.3-0.5     0.8-1.2
        Evening (17-21): 1.2-1.8     1.5-2.2
        Night (22-23):  0.4-0.6     0.5-0.8

    Args:
        n_hours: Number of hours to generate (default: 87600 = 10 years).
        seed: Random seed for reproducibility.
        autocorr: Autocorrelation coefficient (0-1). Higher = smoother transitions.

    Returns:
        np.ndarray: Shape (n_hours,) - load for each hour
    """
    rng = np.random.default_rng(seed)

    # Base loads: [night_early, morning, midday, evening, night_late]
    weekday_base = [0.3, 1.0, 0.4, 1.5, 0.5]
    weekend_base = [0.4, 1.25, 1.0, 1.85, 0.65]

    # Generate continuous timeseries
    loads = np.zeros(n_hours)

    # Generate autocorrelated noise (higher correlation for loads - consumption is sticky)
    noise = _generate_autocorrelated_noise(n_hours, scale=0.15, alpha=autocorr, rng=rng)

    for t in range(n_hours):
        hour_of_day = t % 24
        day_of_week = (t // 24) % 7
        is_weekend = day_of_week >= 5

        base = _get_base_value(hour_of_day, is_weekend, weekday_base, weekend_base)
        loads[t] = base + noise[t]

        # Occasional load spikes (8% chance) - appliances turning on
        if rng.random() < 0.08:
            loads[t] += rng.uniform(0.2, 0.5)

    # Ensure minimum load
    loads = np.maximum(loads, 0.1)

    return loads


def plot_data(prices: np.ndarray, loads: np.ndarray, start_hour: int = 0, n_hours: int = 168):
    """Plot a sample window of price, load, and baseline cost data."""
    prices_slice = prices[start_hour : start_hour + n_hours]
    loads_slice = loads[start_hour : start_hour + n_hours]
    x = np.arange(n_hours)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Price plot
    ax1.plot(x, prices_slice, 'tab:blue', linewidth=1)
    ax1.set_ylabel('Price (EUR/kWh)')
    ax1.set_title(f'Data preview (hours {start_hour}-{start_hour + n_hours})')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.7)

    # Load plot
    ax2.plot(x, loads_slice, color='black', linewidth=1)
    ax2.set_ylabel('Load (kWh)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 3)

    # Accumulated cost plot
    hourly_cost = prices_slice * loads_slice
    accumulated_cost = np.cumsum(hourly_cost)
    ax3.plot(x, accumulated_cost, 'tab:purple', linewidth=1)
    ax3.set_ylabel('Cost (EUR)')
    ax3.set_title(f'Baseline Cost: {accumulated_cost[-1]:.2f} EUR')
    ax3.set_xlabel('Hour')
    ax3.grid(True, alpha=0.3)

    # Set x-axis ticks at day boundaries (multiples of 24)
    n_days = n_hours // 24
    tick_positions = np.arange(0, n_hours + 1, 24)
    ax3.set_xticks(tick_positions)
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    labels = []
    for h in tick_positions:
        abs_hour = start_hour + h
        day_idx = (abs_hour // 24) % 7
        labels.append(f'{h}\n{day_labels[day_idx]}' if h < n_hours else str(h))
    ax3.set_xticklabels(labels)

    # Add vertical lines at day boundaries
    for i in range(n_days + 1):
        for ax in [ax1, ax2, ax3]:
            ax.axvline(i * 24, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "03_data"
    n_hours = 87600  # 10 years * 12 months/year * 730h/months = 87600 hours

    # Generate data
    print(f"Generating {n_hours} hours ({n_hours / 8760:.1f} years) of data...")

    print("\nPrice profiles...")
    prices = generate_price_profile(n_hours=n_hours)
    print(f"  Shape: {prices.shape}")
    print(f"  Range: {prices.min():.3f} - {prices.max():.3f} EUR/kWh")
    print(f"  Mean: {prices.mean():.3f} EUR/kWh")

    print("\nLoad profiles...")
    loads = generate_load_profile(n_hours=n_hours)
    print(f"  Shape: {loads.shape}")
    print(f"  Range: {loads.min():.3f} - {loads.max():.3f} kWh")
    print(f"  Total consumption: {loads.sum():.0f} kWh ({loads.sum() / 10:.0f} kWh/year)")

    # Generate hour-of-day timeseries (0-23 repeating)
    # This is saved alongside price/load so each episode knows the real hour-of-day
    # for each timestep, even when episode_length is not a multiple of 24.
    hours_of_day = np.arange(n_hours) % 24

    # Generate day-of-week timeseries (0=Mon, ..., 6=Sun)
    # Matches the convention in generate_price_profile() where day_of_week >= 5 is weekend.
    days_of_week = np.arange(n_hours) // 24 % 7

    # Save data
    print("\nSaving data...")
    np.save(data_dir / "prices.npy", prices)
    print(f"  Saved: {data_dir / 'prices.npy'}")
    np.save(data_dir / "loads.npy", loads)
    print(f"  Saved: {data_dir / 'loads.npy'}")
    np.save(data_dir / "hours_of_day.npy", hours_of_day)
    print(f"  Saved: {data_dir / 'hours_of_day.npy'}")
    np.save(data_dir / "days_of_week.npy", days_of_week)
    print(f"  Saved: {data_dir / 'days_of_week.npy'}")

    # Plot sample data window
    print("\nPlotting data preview...")
    start_hour = 0  # You can change this to plot a different window (e.g., start_hour=8760 for year 2)
    fig = plot_data(prices, loads, start_hour=start_hour)
    plt.savefig(data_dir / "data_preview.png", dpi=100)
    print(f"  Saved: data_preview.png")

    plt.show()
