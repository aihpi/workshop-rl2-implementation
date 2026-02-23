# Data

Synthetic electricity price and household load profiles at hourly resolution, covering 10 years (87,600 hours).

## Files

| File | Description |
|------|-------------|
| `prices.npy` | Electricity price (EUR/kWh), shape `(87600,)` |
| `loads.npy` | Household load (kWh/h), shape `(87600,)` |
| `hours_of_day.npy` | Hour of day, 0-23, shape `(87600,)` |
| `days_of_week.npy` | Day of week, 0=Mon ... 6=Sun, shape `(87600,)` |
| `data_preview.png` | Plot of the first week |

## Regeneration

```bash
uv run python scripts/generate_data.py
```

### Price profile

- **Weekday/weekend patterns**: higher prices on weekends
- **Daily patterns**: cheap at night (00-06), expensive in the evening (17-21)
- **Autocorrelated noise** (AR(1), alpha=0.7): smooth hour-to-hour transitions
- **Random spikes**: 3% chance per hour of a price spike (+0.05 to +0.15 EUR/kWh)
- **Minimum price**: 0.05 EUR/kWh

### Load profile

- **Weekday/weekend patterns**: higher base load on weekends (people at home)
- **Daily patterns**: low at night, peaks in morning and evening
- **Autocorrelated noise** (AR(1), alpha=0.8): consumption is sticky
- **Appliance spikes**: 8% chance per hour of a load spike (+0.2 to +0.5 kWh)
- **Minimum load**: 0.1 kWh

See `scripts/generate_data.py` for full details.

## Alternative Datasets

The `alternative_datasets/` folder contains data that was explored during development and inspired the design of our synthetic dataset. These are not used in the workshop but can be interesting for further exploration:

- **`demandlib/`** — BDEW H0 standard load profiles via [demandlib](https://github.com/oemof/demandlib). Realistic but too flat (~0.13-0.73 kWh/h) for interesting optimization.
- **`smard/`** — Day-ahead prices and grid load from [SMARD](https://www.smard.de/) (German electricity market). Grid-level data (30,000-80,000 MWh/h), not household-level.

Each folder contains an exploration notebook with visualizations.
