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
Without a key, `nba daily` and `nba odds` use ESPN's free scoreboard for current pregame lines.
With a key, The Odds API is preferred. An invalid or exhausted key is reported; it does not silently
switch sources. When no live snapshots arrive, `nba daily` skips its flag-writing step.

The default database is `data/nba.db`. Set `NBA_DB_PATH` to an alternate path for experiments.
Settings use the `NBA_` environment prefix, including `NBA_CURRENT_SEASON` (default `2026-27`),
`NBA_NBA_DELAY_S` (0.6) and `NBA_ESPN_DELAY_S` (1.0). `ODDS_API_KEY` is unprefixed.
Dates are US/Eastern calendar dates; stored tip times are UTC.

By default, model features, Elo history and fitting use **2025–26 and 2026–27 only**
(`NBA_MODEL_SEASONS='["2025-26","2026-27"]'`). Older data remains available in Explore and
Advanced data pages. Daily predictions fit on eligible games strictly before the selected date;
the default backtest fits on 2025–26 and evaluates 2026–27 with a frozen fit.

## First backfill

Ingest stats first, then match ESPN events and lines. All commands below preserve existing data;
repeating an ingest updates matching rows. Do not remove the database to resume a partial run.

```sh
uv run nba init-db
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
| `nba features` | Seconds to a minute, depending on the stored history |

ESPN waits at least one second before each request, so a full season with roughly 1,230 games plus
scoreboards and fallback calls takes tens of minutes before network latency. Reruns skip completed
work and are much faster. Missing summaries can return 404/410; these are skipped, while transient
failures remain retryable. `--refetch` intentionally re-requests data.

Optional older-season line recovery and opening-line collection:

```sh
uv run nba ingest --season 2023-24
uv run nba ingest --season 2024-25
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

# Before tip: refresh the current lines and save forecasts
uv run nba odds
uv run nba predict

# Next morning, after updating final scores
uv run nba daily
uv run nba grade
```

With an Odds API key, `uv run nba odds --closing` marks the captured snapshot as closing; it does
not schedule a pull or guarantee the final price for later games. Without a key, ESPN scoreboard
snapshots remain current observations even with `--closing`; finished-game ESPN ingest supplies
historical closing lines. Only finished games are graded. Predictions made after tip and games
already started are excluded from new live flags. The Advanced Today page retains the timing groups
and per-game model playground.

For model evaluation, run the default last-season backtest (allow several minutes). Use
`--all-seasons` explicitly for the older monthly-refit analysis; that opt-in also exists on
`nba predict`. Holdout seasons are reported separately and do not select the preferred model.

```sh
uv run nba backtest
# Optional all-history analysis:
uv run nba backtest --all-seasons --season 2024-25 --season 2025-26 --holdout 2026-27
uv run pytest -q
```

## Dashboard

```sh
uv run streamlit run app/dashboard.py
```

Open the local URL printed by Streamlit. The main navigation is **Game Brief**, **Results Review**
and **Explore**. Existing technical pages are under **Advanced**.

1. **Game Brief** opens at today's Eastern date. Read the slate, choose a game, then compare its
   market line and source, projected home margin and total, spread evidence, total evidence and
   pace, moneyline probabilities, last 10 results for each team, and last 3 head-to-head meetings.
   Missing lines and prices are shown explicitly. First-observed/current comparisons use the same
   source and provider; the first stored observation may not be the actual opening line.
2. Change **Trend split** to all games, home, road, favourite or underdog. Each table shows last
   5/10/15 and season records, decided-game counts, pushes and missing lines. The evidence checks
   always use overall and upcoming-venue last 10. All trend inputs precede the chosen game date.
3. Open **Results Review** after refreshing final scores. Filter by season, dates, teams and
   prediction type to compare model projections, closing lines and final scores. Review who covered,
   whether the total went over or under, and whether each model direction hit. Choose a team for
   its last-5/10/15 cover record through the range end; export the readable results as CSV.
4. Use **Explore → ATS & O/U by team** for cover and over/under records and rates with the existing
   filters. **Advanced** contains Today / model playground, Team Trends, Data, SQL, Flag Log,
   Backtest, Custom Features, Line Movement and Data Health.

**Reading tendencies:** direction follows the model's difference from the market. One supporting
check each comes from a model difference of at least 2 spread points / 3 total points / 3 percentage
points for moneyline, both teams' last-10 trends, and both teams' last-10 venue trends. Trend support
needs at least 55% in the chosen direction (opponent at most 45% for spread/moneyline) and 5 decided
games per team. Three checks = Strong, two = Lean, one = Slight, zero = None. These labels describe
evidence, not calibrated confidence. Tables show the checks and their sample sizes.

**Forecast provenance:** a saved forecast counts as recorded before tip only with verified timing,
training dates and the configured season scope. Otherwise the app reconstructs a **retro** estimate
using only earlier game dates, without saving or replacing predictions. Upcoming estimates can be
marked **Preview · not saved** when the tip time is known. Hit rates always separate recorded and
retro forecasts; pushes and missing evidence are excluded. Fewer than 30 decided games is labelled
**small sample**. Game Brief's honesty check also offers a separately labelled **2025–26 in-sample**
comparison, fitted on that same season; it is not independent validation. Results Review always uses
pre-date estimates, including when reviewing last season.

The pages use stored local data and may refresh derived tables; opening them does not fetch new
scores or lines. Run the CLI refresh first. The first season review takes longer while retro
estimates are computed; subsequent views reuse cached results until the database or model settings change.

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
  Current scoreboard snapshots use real observation times and may omit lines or prices.
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
