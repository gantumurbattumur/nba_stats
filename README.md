# NBA stats & odds analysis

A local data-engineering showcase: NBA team-game logs and market snapshots flow into SQLite,
derived metrics, leakage-aware features, Elo/window/ML predictions, and a Streamlit dashboard.
Every stage can be inspected with filters, SQL and CSV exports.

**Analysis only — no bets placed. No account connections.** Value flags record model-versus-market
disagreements. Returns and CLV are model-evaluation metrics only; this project does not execute transactions.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run these commands from the repository root.
The repository pins Python 3.12; use `uv run` for Python and CLI commands.

```sh
uv sync
cp .env.example .env
uv run nba init-db
```

Get a free data API key at [The Odds API](https://the-odds-api.com/) and set
`ODDS_API_KEY=your_key` in `.env`. Keep `.env` private; it is ignored by Git.
Without a key, historical stats, ESPN lines and predictions still work. `nba daily` reports
unavailable live odds and skips its flag-writing step if the key is missing, invalid or out of quota.

The default database is `data/nba.db`. Set `NBA_DB_PATH` to an alternate path for experiments.
Settings use the `NBA_` environment prefix, including `NBA_CURRENT_SEASON` (default `2026-27`),
`NBA_NBA_DELAY_S` (0.6) and `NBA_ESPN_DELAY_S` (1.0). `ODDS_API_KEY` is unprefixed.
Dates are US/Eastern calendar dates; stored tip times are UTC.

## First backfill

Ingest stats first, then match ESPN events and lines. All commands below preserve existing data;
repeating an ingest updates matching rows. Do not remove the database to resume a partial run.

```sh
uv run nba init-db
uv run nba ingest --season 2023-24
uv run nba ingest --season 2024-25
uv run nba ingest --season 2025-26
uv run nba ingest --season 2026-27

uv run nba espn --from 2025-10-21 --to 2026-04-12
uv run nba espn --from 2026-04-13 --to 2026-06-13
uv run nba espn --from 2026-10-20 --to 2026-11-14

uv run nba features
uv run nba status
```

The ranges above are the project's backfill snapshot as of November 15, 2026. Extend the current
season's end date through yesterday when running later. Ingest defaults to regular season, play-in
and playoffs; use `--season-type regular` to restrict it. ESPN processes final games already in the
database, so ingesting logs first matters.

Planning estimates for a first run, rather than guaranteed benchmarks:

| Step | Expected time |
| --- | --- |
| `nba init-db` | A few seconds |
| `nba ingest --season ...` | About 10–60 seconds per season on a responsive endpoint; retries can take several minutes |
| ESPN full regular season | About 30–90 minutes, including per-game summaries and core-API fallbacks |
| ESPN play-in/playoffs | About 3–15 minutes |
| ESPN current season through November 14 | About 5–15 minutes |
| `nba features` | Seconds to a minute for the four-season dataset |

ESPN waits at least one second before each request, so a full season with roughly 1,230 games plus
scoreboards and fallback calls takes tens of minutes before network latency. Reruns skip completed
work and are much faster. Missing summaries can return 404/410; these are skipped, while transient
failures remain retryable. `--refetch` intentionally re-requests data.

Optional older-season line recovery and opening-line collection:

```sh
uv run nba espn --from 2024-10-22 --to 2025-06-22
uv run nba espn --from 2023-10-24 --to 2024-06-17
uv run nba espn --from 2025-10-21 --to 2026-06-13 --with-open
```

Older ranges can take 45–120 minutes because empty summaries trigger an additional core request
per game. Coverage varies; inspect `nba status` and Data Health afterward. Core requests are tracked
per game, so repeats skip previously checked events unless explicitly refetched.

## Daily routine

```sh
# Morning: update logs, slate, yesterday's ESPN lines, live odds, features and predictions
uv run nba daily

# About 30 minutes before the first tip: mark a fresh live snapshot as closing
uv run nba odds --closing

# Next morning, after updating final scores
uv run nba daily
uv run nba grade
```

`--closing` labels the snapshot you capture; it does not schedule a pull or guarantee the final
market price for later games. Additional pulls nearer later tips consume quota. Only finished games
are graded. Predictions made after tip and games already started are excluded from new live flags.
Today separates pre-tip predictions, late predictions, unknown timing, and final games.

For model evaluation, run the walk-forward backtest (allow several minutes; runtime depends on the
machine and ML refits). The holdout is reported separately and does not select the preferred model.

```sh
uv run nba backtest --season 2024-25 --season 2025-26 --holdout 2026-27
uv run pytest -q
```

## Dashboard

```sh
uv run streamlit run app/dashboard.py
```

Open the local URL printed by Streamlit. Pages cover Today, Team Trends, Explorer, Data, SQL,
Flag Log, Backtest, Custom Features, Line Movement and Data Health.

Today's per-game playground supports any positive window N, normalized window weights, venue splits,
home-court/rest adjustments, SOS and separate input filters for each team. It recomputes the window
model and market comparisons live and shows the profile and game rows used, strictly before the
selected date. Playground changes are temporary. Short histories use available games; season-to-date
stays within the current season, while last-N windows may reach into previous history early in a season.

Custom Features accepts named SQL expressions over `game_features`, with a 20-row preview and a
distribution over all values. Save validates the expression through the guarded read-only query path.
Enabled features reach ML on the next backtest/predict; existing results remain unchanged. Use only
pre-game inputs: expression validation prevents writes, but cannot prove a feature is free of future
information. The SQL page also runs through the read-only query guard.

## Data sources and limitations

- **NBA stats:** `nba_api` is an unofficial client of NBA endpoints. Availability, schemas and rate
  limits can change. The client delays calls and retries transient failures; a blocked endpoint may
  still require retrying later. Source box scores can contain anomalies; inspect Data Health.
- **ESPN:** unofficial scoreboard, summary and core endpoints. The initial summary `pickcenter`
  field is empty for the project's 2023–25 samples, even though the core API has 2024–25 lines.
  The ingester already falls back to core odds. The checked local 2025–26 regular season has closing
  lines for all 1,230 games; this does not promise complete postseason, older-season or opening-line
  coverage. Historical ESPN timestamps are synthetic open/close markers, not observed pull times.
- **No injury data:** models omit injuries, expected lineups and player availability.
- **Market interpretation:** spreads are always from the home team's perspective (negative means
  home favored). Missing prices remain missing. Market comparisons use available snapshots and
  no-vig probabilities; they are not a guarantee of predictive quality.
- **Odds API:** this project requests only current NBA odds at
  `GET https://api.the-odds-api.com/v4/sports/basketball_nba/odds`, with `regions=us`,
  `markets=h2h,spreads,totals`, and `oddsFormat=american`. It never calls the historical endpoint.

The [free plan](https://the-odds-api.com/) currently provides 500 credits per month. Under the
[v4 usage rules](https://the-odds-api.com/liveapi/guides/v4/), three markets × one region normally
costs **3 credits per nonempty pull**. One morning pull plus one closing pull is 6 credits/day:
180 credits over 30 days, or 186 over 31. At 3 credits each, 500 credits allows 166 such pulls.
Repeated `nba daily` calls also pull odds. Check actual usage in `api_usage`; empty responses and
provider billing rules can affect the charge. Plan details were checked November 15, 2026.
