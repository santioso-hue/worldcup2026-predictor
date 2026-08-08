# worldcup2026-predictor

Win probabilities for the 2026 World Cup (48 teams, 12 groups), recomputed from real
results as the tournament played out. Elo ratings fitted on the full history of
international matches set team strengths, a Dixon-Coles model turns those into goal
distributions, and a Monte Carlo simulation plays out whatever is left of the
tournament. The model ran live from the group stage to the final; the dashboard now
shows the finished tournament.

## Getting started

```bash
make setup                        # creates the .venv and installs dependencies
make run                          # full live run (needs FOOTBALL_DATA_TOKEN)
streamlit run app/dashboard.py    # interactive dashboard (reads the latest run)
```

Without an API key you can still simulate the pre-tournament baseline, which pulls the
schedule (openfootball) and historical results (martj42):

```bash
python scripts/run_pipeline.py --mode pre_tournament --runs 50000
```

## Commands

```bash
# Forecast a single match (1X2), using martj42 history.
# Names must match martj42's spelling (e.g. "United States", not "USA").
python scripts/predict_match.py "Brazil" "France"
python scripts/predict_match.py "United States" "Mexico" --host "United States"

# Reproduce an exact state (numbers won't change)
python scripts/run_pipeline.py --snapshot 20260616t1830 --runs 50000

# Keep refreshing while matches are in progress (polls in windows; see triggers.py)
python scripts/run_pipeline.py --watch --interval 600

# Quality checks (all run inside .venv, via make)
make test                         # pytest
make lint                         # ruff + black --check + mypy
make fmt                          # black + ruff --fix
```

During the tournament, the
[`.github/workflows/refresh.yml`](.github/workflows/refresh.yml) workflow ran the
pipeline every 30 minutes via GitHub Actions and uploaded the figures and JSON as
artifacts; it is manual-dispatch only now.

## Output

- `data/processed/probabilities_<ts>.json` (and `latest.json`): per team, the
  probability of advancing and of reaching the round of 16, quarterfinals, semis,
  final, and title, plus group standings. The `latest.json` pointer drives the ↑/↓
  arrows against the previous run.
- `outputs/figures/*.png`: champion ranking, 1X2 bar chart, scoreline heatmap, bracket,
  group tables, and reliability curve (1080x1920 vertical or 1920x1080 horizontal).
- `outputs/videos/*.mp4`: the animation that reveals the champion.
- `data/raw/results_<ts>.parquet`: the exact inputs the run saw, kept so any past
  state can be re-run.

Runs are reproducible: the same snapshot and seed give identical output. In `live`
mode the figures change as new results come in (that's expected); to pin a state,
use `--snapshot <ts>`.

## How it works

The pipeline runs in four stages:

- **Live data:** `FootballDataProvider` pulls results; the schedule comes from
  openfootball and history from martj42. Each run saves a timestamped snapshot,
  validated and reconciled.
- **Dynamic Elo:** sequential, deterministic fit over history, with a goal-margin
  multiplier (eloratings.net) and recency weighting.
- **Dixon-Coles:** converts the Elo gap into expected goals, builds the Poisson matrix,
  and corrects low scorelines with τ.
- **Conditional Monte Carlo:** Annex C, Article 13 tiebreakers (recursive
  head-to-head), extra time and penalties; simulates only what's left to play and
  outputs P(round/title).

The backtest (walk-forward, with Platt calibration), the charts (PNG/MP4), and the
Streamlit dashboard all read from the artifacts each run leaves behind.

## Architecture

Layers are separated and dependencies flow one way:
`data → features → models → simulation → evaluation → viz`. I/O lives in `data`,
`scripts`, and `app`. All randomness goes through `worldcup.rng.get_rng()`, so the same
seed always reproduces the same output.
