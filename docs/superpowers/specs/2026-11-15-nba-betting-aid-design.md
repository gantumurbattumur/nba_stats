# NBA Betting Aid — Design & Implementation Plan

## Context
The repo (`/Users/ganabattumur/GANA/projects/nba_stats`) is empty, with no commits. The user wants a **personal betting aid** that runs locally. Each day it pulls recent NBA stats and today's sportsbook odds, predicts every game on today's slate, and shows where the model disagrees with the market. The predictions draw on recent form (last 5/8/10/15 games), home/away splits and other standard methods. The chosen approach is **Python + a local SQLite database + a Streamlit dashboard**.

Success means:
1. Each morning, one command gives a ranked list of value picks.
2. The dashboard lets the user check why a pick looks good (trends, splits, rest).
3. A pick log shows whether the model is actually any better than the lines (hit rate, ROI, closing-line value).

## Goals / Non-goals / Anti-goals
- **Goals:** daily predictions for moneyline, spread and total; model win chance vs no-vig market chance; trend views by window and by home/away; a pick log with grading and CLV; a backtest that compares model variants.
- **Non-goals (v1):** player-level props, injury feeds (no free reliable source; the dashboard notes this), hosting, multiple users, live in-game odds.
- **Anti-goals:** data leakage (features must use only games *before* the game being predicted); unearned confidence (always show calibration and backtest numbers next to picks); burning the free odds quota.

## Constraints and data sources
- **Stats: `nba_api`** (free, unofficial wrapper of stats.nba.com). It works locally but blocks cloud IPs, and that doesn't matter here.
  - `leaguegamelog.LeagueGameLog(season, player_or_team_abbreviation='T')` returns every team box score for a season in one call. Possessions are estimated as `FGA + 0.44·FTA − OREB + TOV`, which avoids pulling advanced stats game by game.
  - Today's slate comes from `nba_api.live.nba.endpoints.scoreboard` (or `scoreboardv2` for a specific date).
  - Add about 0.6 s between calls, plus retries and browser-like headers.
- **Odds: The Odds API v4** (free tier: about 500 credits a month).
  - Endpoint: `GET /v4/sports/basketball_nba/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american`, which costs 3 credits per call.
  - Two pulls a day (morning, and near tip-off for the closing line) is about 180 credits a month, inside the limit.
  - The key lives in `.env` as `ODDS_API_KEY`. The user signs up for it.
  - The Odds API's historical endpoint is paid only, so it's used **only for today's live odds**.
- **Past-game odds: ESPN's unofficial API** (free, no key). Verified 2026-11-15:
  - `site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD` lists event ids.
  - `…/summary?event=ID` → `pickcenter[0]` contains `spread`, `overUnder`, `overOdds`/`underOdds`, and `homeTeamOdds`/`awayTeamOdds.moneyLine`/`spreadOdds`.
  - Coverage: the **DraftKings** line for 2025-26 and 2026-27; a "consensus" line for 2022-23 and earlier; **nothing for 2023-24 or 2024-25**.
  - That gives every team's last 5/10/15 games with lines, all of last season, and a real against-the-odds backtest on 2025-26.
  - It costs one call per game (about 1230 a season). Backfill once at about 1 request per second and cache it. After that, pull only new games each day.
  - The data is unofficial, so the parser must cope with missing fields. Games with no line are stored with no odds, not dropped.
- Seasons: the current season is 2026-27. Results-based training uses 2023-24 to 2025-26. Odds-based backtests and records against the spread / over-under use 2025-26 and 2026-27.

## Approaches considered
| | Approach | Trade-off |
|---|---|---|
| **Chosen** | Python + SQLite + Streamlit | One language, local, fast to build, interactive trends |
| | CLI + Jupyter notebooks | Least work, clunky day to day |
| | FastAPI + React | Polished, but two codebases; only worth it for a public site |

## Prediction methods (my call; the backtest chooses the final blend)
1. **Elo:** the standard NBA setup (K≈20, home advantage about 100 Elo points, margin-of-victory multiplier, seasons pulled back toward the mean). It serves as the baseline.
2. **Window rating model** (the user's core ask):
   - Per team, computed on games before the date: net rating, offensive and defensive rating, pace, point differential and win % over **L5, L8, L10, L15 and the season so far**.
   - Each of those is also computed on **home-only and away-only** subsets: the home team's last N home games, the away team's last N road games.
   - Adjust for opponent strength: the average net rating of the opponents faced in each window.
   - Situational factors: rest days, back-to-back, 3 games in 4 nights, rest advantage.
   - Predicted margin = blended rating gap + home-court advantage + rest adjustments. Window weights and home-court advantage are fit in the backtest.
   - Win chance = Φ(margin / σ), with σ fit (about 12–13 points).
   - Predicted total = expected pace × the two teams' offense-vs-defense matchup.
3. **ML model:** scikit-learn logistic regression for winning and ridge regression for margin and total.
   - Inputs are home-minus-away differences of the features above, plus Elo.
   - Validation goes forward through time, never on shuffled games, and win chances are calibrated.
   - Odds-based inputs from the ESPN lines, per window and per home/away split: record against the spread, over/under record, average margin vs the spread, and whether the team was favored.
4. **Market comparison:**
   - Convert American odds to implied chance, then strip the bookmaker's margin by scaling both sides to sum to 1. Use the consensus across books, plus the best available price.
   - Edge = model chance − market chance. For spreads and totals, edge = model margin or total − the line.
   - Flag a pick when edge ≥ 3% (configurable). Suggest a stake at ¼ Kelly of a configurable bankroll.

## Filters and custom analysis (user request: "any sort of filters", "see any custom data engineering results")
Every table the pipeline produces stays queryable. Nothing is hidden behind fixed charts.
- **One filter engine** (`analysis/filters.py`) shared by the dashboard, the CLI and the models.
  - A `FilterSpec` (pydantic) turns into SQL/pandas. It covers:
    - season(s) and date range
    - team(s) and opponent(s)
    - home/away
    - **last N games, for any N** (not just 5/8/10/15)
    - days of rest and back-to-back
    - favorite or underdog, spread range, total range
    - result: win/loss, covered or not, over/under
    - opponent strength (e.g. opponents with net rating in the top 10 or above .500)
    - conference/division, month, day of week
  - Specs can be saved as named **presets** in SQLite.
- **Explorer page:**
  - filters + a picker for any metric (raw box score or derived: offensive/defensive/net rating, pace, effective FG%, turnover rate, offensive rebound rate, free-throw rate, record vs the spread, over/under record, margin vs the spread)
  - an aggregation (mean/median/sum/record) and a group-by (team, opponent, home/away, rest bucket, month, favorite/underdog)
  - shown as a table and a chart, with a CSV download
- **Data page:** browse every stage of the pipeline: raw game logs, odds snapshots (ESPN past + Odds API live), the per-game feature table, predictions, picks. Each view shows the same filters and row counts.
- **SQL console:** run any SELECT against the database through a **read-only** connection (`file:…?mode=ro`), with results shown as a table, a chart and a CSV.
- **Custom features:**
  - The user defines a named column with a SQL expression over the feature table, e.g. `l5_net_home - l10_net_away`.
  - It's stored in a `custom_features` table and appears in the Explorer.
  - It can be switched on as an ML input. The backtest page then shows whether it helped.
- **Model playground (Today page):** sliders and pickers for windows (any N), weights, home-court advantage, rest adjustment, and a FilterSpec applied to each team's input games (e.g. "only games vs above-.500 teams"). Predictions and edges recompute instantly; it's the window model, so there's no retraining.

## Project structure
```
nba_stats/
  pyproject.toml            # uv; deps: nba_api, pandas, numpy, scipy, scikit-learn,
                            # requests, typer, pydantic-settings, streamlit, plotly, pytest
  .env.example  .gitignore  # ignores data/*.db, .env
  README.md
  docs/superpowers/specs/2026-11-15-nba-betting-aid-design.md   # this design, committed
  src/nba_predict/
    config.py               # settings (env + defaults: windows, edge threshold, kelly frac, bankroll)
    db.py                   # SQLite schema + helpers
    data/nba_client.py      # nba_api wrapper: rate limit, retries, season game logs, today's slate
    data/odds_client.py     # Odds API client, credit-usage logging
    data/espn_client.py     # past-game lines from ESPN summary/pickcenter (cached, tolerant parser)
    data/teams.py           # map NBA team ids ↔ Odds API names ↔ ESPN team ids
    analysis/filters.py     # FilterSpec → SQL/pandas; presets
    analysis/metrics.py     # derived metric registry (ratings, four factors, ATS/O-U)
    analysis/custom.py      # custom feature definitions (validated SQL expressions)
    ingest.py               # backfill + incremental update of game logs; odds snapshots
    features/rolling.py     # L5/8/10/15/season windows, home/away splits, possessions/ratings
    features/schedule.py    # rest, b2b, 3-in-4
    features/build.py       # one leakage-safe feature row per game (as-of date)
    models/elo.py
    models/window_rating.py
    models/ml.py
    odds_math.py            # american↔prob, devig, edge, kelly
    predict.py              # today's pipeline → predictions table
    picks.py                # flag value picks, log them, grade results, CLV vs closing snapshot
    backtest.py             # walk-forward evaluation of all models
    cli.py                  # typer: ingest | odds | predict | grade | backtest | daily
  app/dashboard.py          # Streamlit multipage entry
  app/pages/                # Today(+playground) · Team trends · Explorer · Data · SQL · Pick log · Backtest
  tests/                    # pytest with small fixture game logs
  data/                     # nba.db (gitignored)
```
**SQLite tables:** `team_games` (one row per team per game, with the box score), `games`, `odds_snapshots` (source espn|oddsapi, book, market, price/line, pulled_at, is_closing), `filter_presets`, `custom_features`, `predictions` (model, game, prob/margin/total, created_at), `picks` (market, side, line, price, edge, stake, result, clv).

## Implementation order (test-first for logic pieces)
1. **Scaffold.** Set up uv, `pyproject`, config and `.gitignore`. Save this design to `docs/superpowers/specs/…` and make the first commit.
2. **Odds math** (`odds_math.py`) with unit tests: odds conversion, devig, edge, Kelly.
3. **Ingest.**
   - `nba_client` + `db`. `nba ingest --seasons 2023-24..2026-27` backfills; running it again pulls only new games.
   - Today's slate from the live scoreboard.
   - ESPN line backfill for 2025-26 and 2026-27 (optionally ≤2022-23) is resumable, and runs again only for games it hasn't fetched.
4. **Features + filter engine.** Rolling windows, splits, opponent adjustment, schedule features and ATS/O-U records. Also `FilterSpec` and the metric registry, which the features reuse.
   - Tests use hand-checked fixtures.
   - Include a **leakage test**: changing or removing a game on or after date D must not change features as of D.
5. **Models + backtest.**
   - Build Elo, the window rating model and the ML model.
   - `nba backtest` runs walk-forward over 2024-25 and 2025-26 and reports accuracy, Brier score, log loss, margin MAE and total MAE per model, against an "always pick home" baseline.
  - On 2025-26 it also scores against the market using ESPN lines: hit rate and ROI on flagged picks, and model log loss vs the market's no-vig log loss.
   - Fit the window weights, home-court advantage and σ here.
6. **Odds ingest + team-name mapping.** `nba odds` stores a snapshot and logs credits remaining. A `--closing` flag marks the near-tip pull.
7. **Predict + picks.**
   - `nba predict` writes predictions for today's games with the best model or blend.
   - `picks.py` flags edges and suggests stakes.
   - `nba grade` settles finished games and computes CLV.
   - `nba daily` = ingest → odds → predict → flag.
8. **Dashboard.** All pages use the shared filter sidebar. Pages:
   - **Explorer, Data, SQL console** and the custom-feature editor, as described above.
   - **Model playground** on the Today page.
   - **Today:** game cards with model vs market chance, edge, projected spread and total vs line, and rest/b2b flags.
   - **Team trends:** a team picker, net rating by window, home vs away.
   - **Pick log:** record, ROI, CLV, a calibration chart.
   - **Backtest:** the metrics table.
9. **README.** Setup, getting the API key, daily routine, and how to schedule `nba daily` with launchd/cron (scheduling is optional).

## Left to the builder
Exact Elo parameters, window-blend weights (fit in the backtest), ridge/logistic regularization, the dashboard's visual details, and whether to blend models or pick the single best one (decided by backtest log loss).

## Verification
- `uv run pytest`: odds math, window calculations against hand-computed fixtures, the leakage test, grading logic.
- `uv run nba ingest --seasons 2023-24 2024-25 2025-26 2026-27`: row counts are about 1230 games × 2 per full season.
- `uv run nba backtest`: every model beats the always-home baseline on log loss, and win chances look calibrated.
- With a real `ODDS_API_KEY`: `uv run nba daily` prints today's slate with edges, and credits used per call are logged (should be 3).
- `uv run streamlit run app/dashboard.py`: click through every page.
  - Explorer filter check: "MIL, away, last 7, vs above-.500" shows exactly the matching games, and the numbers match a hand-written SQL query in the SQL console.
  - The SQL console rejects writes (e.g. `DELETE`).
  - A custom feature shows up in the Explorer and in the backtest.
- `uv run pytest` also covers: FilterSpec → SQL for each filter type, and the ESPN parser on a saved summary JSON with and without `pickcenter`.
- Next day: `uv run nba grade` settles yesterday's picks, and the pick log updates.
