# NBA Stats & Odds Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local tool that ingests NBA stats and odds, predicts today's games from recent form/splits/rest/Elo, flags model-vs-market disagreements, logs and grades those flags, and exposes every data stage through free-form filters, SQL and custom features in a Streamlit dashboard.

**Architecture:** One Python package `nba_predict` (src layout) writing to one SQLite file. Pipeline stages each materialize a table (`games`/`team_games` → `team_games_enriched` → `game_features` → `predictions` → `value_flags`) so the dashboard can show any stage. One `FilterSpec` drives the dashboard, CLI and model playground.

**Tech Stack:** Python 3.12 (uv), nba_api, requests, pandas, numpy, scipy, scikit-learn, pydantic / pydantic-settings, typer, streamlit, plotly, pytest, responses (HTTP mocking).

**Spec:** `docs/superpowers/specs/2026-11-15-nba-betting-aid-design.md`

## Global Constraints

- Python `>=3.12` via `uv python pin 3.12` (system python is 3.9 — never use it). Run everything with `uv run`.
- Package `nba_predict` under `src/`; CLI entry point `nba = "nba_predict.cli:app"`.
- DB path from setting `db_path`, default `data/nba.db`; `data/*.db` and `.env` are gitignored.
- Settings (pydantic-settings, env prefix `NBA_`, plus `ODDS_API_KEY` unprefixed): `windows=(5, 8, 10, 15)`, `edge_threshold=0.03`, `current_season="2026-27"`, `train_seasons=("2023-24","2024-25","2025-26")`, `nba_delay_s=0.6`, `espn_delay_s=1.0`.
- **Analysis only — data-engineering showcase.** No bankroll, stake sizing, Kelly, bet placement, or connection to any bank, wallet or sportsbook account, anywhere (code, settings, UI, docs). "Value flags" are model-vs-market disagreements; "1-unit ROI" is a model-evaluation metric only.
- Odds API call is exactly `GET https://api.the-odds-api.com/v4/sports/basketball_nba/odds` with `regions=us`, `markets=h2h,spreads,totals`, `oddsFormat=american`. Never call the historical endpoint.
- ESPN: `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD` and `.../summary?event=ID`, lines from `pickcenter[0]`. ESPN `spread` is from the **home** team's perspective (negative = home favored).
- No leakage: any feature for a game on date D uses only games with `game_date < D`.
- All dates are US/Eastern calendar dates stored as `YYYY-MM-DD` text.
- Spread convention everywhere: `home_spread` (home perspective). Team-perspective spread = `home_spread` if home else `-home_spread`.
- Any window N is allowed anywhere; `(5,8,10,15)` are only defaults. `szn` = season-to-date.
- User-supplied SQL (console and custom features) runs only through `db.readonly_query`.
- Commit after every task; messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Agent Workflow (user-directed, 2026-11-15)

Three agents work on branch `dev`, in four phases. Phase 1 = Tasks 1–4, Phase 2 = Tasks 5–10, Phase 3 = Tasks 11–13, Phase 4 = Tasks 14–16.

| Agent | Role | Writes | Never |
|---|---|---|---|
| **A — Builder** | Implements the phase's tasks test-first, one commit per task. At the start of each phase after the first, reviews C's fixes from the previous phase. | `src/`, `tests/`, `app/`, root config | edits reports of B/C |
| **B — Analyst** | Runs the real pipeline for what exists (ingest, ESPN backfill, features, backtest, explore), sanity-checks the data (row counts, home win rate, spread sign, line coverage, leakage spot checks), writes findings with the exact SQL/commands used. Reports bugs to C. | `docs/reports/phase-N-B-analysis.md`, `data/` | edits `src/`/`tests/`; commits |
| **C — Tester/Debugger** | Runs the full suite, reviews A's diff against this plan, adds missing edge-case tests, fixes bugs (incl. B's), re-runs 2–3 of B's queries to confirm B's numbers. | `tests/`, fixes in `src/`/`app/` (commits `fix:`/`test:`), `docs/reports/phase-N-C-review.md` | edits B's report |

Cross-checks: C reviews A's code · A reviews C's fixes · C verifies B's numbers · B's data findings feed C's debugging. The coordinator commits reports.

**Every agent report ends with a `## Next` section holding at least one concrete suggested new feature or one bug fix.**

## Review Focus

1. **Too few prior games** (season start, or N larger than games played): stats compute over the games available, a `gp` count is shown, nothing crashes or becomes all-NaN. → tests in Task 7 (`test_short_history_uses_available_games`) and Task 9 (`test_predict_with_two_prior_games`).
2. **Game with no odds** (ESPN gap 2023-25, Odds API missing a game, unmapped team name): prediction still produced and shown, edge columns empty, nothing flagged. → Task 4 (`test_summary_without_pickcenter`), Task 12 (`test_unmapped_team_skipped_with_warning`), Task 13 (`test_no_odds_no_flag`).
3. **Odds API key missing/invalid/quota exhausted** (missing key, 401, 429): `nba daily` prints a clear message and still writes predictions. → Task 12 (`test_quota_exhausted_raises_clean_error`), Task 13 (`test_daily_without_odds_key`).
4. **UTC vs Eastern date boundary and postponed games**: a 10:30pm ET tip (next-day UTC) matches the right game; postponed/unfinished games are never graded. → Task 12 (`test_late_tip_matches_eastern_date`), Task 13 (`test_grade_skips_unfinished`).
5. **Filter that matches nothing**: Explorer/Data/aggregate return empty frames and the page shows an empty-state message. → Task 6 (`test_empty_result`), Task 15 (AppTest with impossible filter).

---

## File Map

```
pyproject.toml, .python-version, .env.example, .gitignore, README.md
src/nba_predict/
  config.py          Settings
  db.py              connect(), init_schema(), upsert(), readonly_query()
  teams.py           TEAMS table: nba_id, abbr, name, odds_name, espn_abbr, conf, div
  odds_math.py       pure odds functions
  data/nba_client.py nba_api wrapper (rate limit + retry)
  data/espn_client.py ESPN scoreboard/summary fetch + parse
  data/odds_client.py Odds API fetch + parse
  ingest.py          ingest_season, ingest_slate, ingest_espn_lines, ingest_odds
  analysis/metrics.py  per-game derived columns + METRICS registry + aggregate()
  analysis/enrich.py   build_enriched() -> team_games_enriched
  analysis/filters.py  FilterSpec, apply_filter(), presets
  analysis/custom.py   custom feature CRUD + evaluation
  features/build.py    build_feature_table(), team_profile()
  models/elo.py, models/window_rating.py, models/ml.py
  backtest.py, predict.py, flags.py, cli.py
app/dashboard.py, app/components/{data.py,sidebar.py}, app/pages/*.py
tests/conftest.py (fixture DB factory), tests/fixtures/*.json, tests/test_*.py
```

### Schema (Task 1 creates all of it)

| table | columns (PK first) |
|---|---|
| `games` | `game_id` TEXT PK, `season`, `game_date`, `home_team_id` INT, `away_team_id` INT, `home_pts` INT NULL, `away_pts` INT NULL, `status` TEXT (`scheduled`/`final`/`postponed`), `espn_event_id` TEXT NULL |
| `team_games` | PK(`game_id`,`team_id`), `opp_team_id`, `season`, `game_date`, `is_home` INT, `wl`, `min`, `pts`, `fgm`, `fga`, `fg3m`, `fg3a`, `ftm`, `fta`, `oreb`, `dreb`, `reb`, `ast`, `stl`, `blk`, `tov`, `pf`, `plus_minus` |
| `odds_snapshots` | `id` INT PK AUTOINC, `game_id`, `source` (`espn`/`oddsapi`), `book`, `pulled_at` ISO UTC, `is_closing` INT, `home_ml`, `away_ml`, `home_spread`, `home_spread_price`, `away_spread_price`, `total`, `over_price`, `under_price` (all nullable); UNIQUE(`game_id`,`source`,`book`,`pulled_at`) |
| `api_usage` | `id` PK, `at`, `source`, `credits_used` INT, `credits_remaining` INT |
| `predictions` | `id` PK, `game_id`, `model`, `created_at`, `home_win_prob`, `pred_margin` (home−away), `pred_total`, `params_json`; UNIQUE(`game_id`,`model`) — re-predict replaces |
| `value_flags` | `id` PK, `game_id`, `created_at`, `model`, `market` (`ml`/`spread`/`total`), `side` (`home`/`away`/`over`/`under`), `line` NULL, `price` INT, `book`, `model_prob`, `market_prob`, `edge`, `result` NULL (`win`/`loss`/`push`), `units` NULL (1-unit return, evaluation only), `closing_line` NULL, `closing_price` NULL, `clv` NULL; UNIQUE(`game_id`,`market`,`side`) |
| `filter_presets` | `name` PK, `spec_json`, `created_at` |
| `custom_features` | `name` PK, `expression`, `use_in_model` INT, `created_at` |
| `backtest_results` | `run_at`, `model`, `season`, `metric`, `value` |
| `team_games_enriched`, `game_features` | written by pandas `to_sql(if_exists="replace")` in Tasks 5 and 7 |

Odds are stored wide (one row per book per pull) rather than long per market — my call; one row maps 1:1 to an ESPN `pickcenter` entry and an Odds API bookmaker.

---

### Task 1: Scaffold, settings, schema, teams

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `.env.example`, `src/nba_predict/__init__.py`, `src/nba_predict/config.py`, `src/nba_predict/db.py`, `src/nba_predict/teams.py`, `tests/conftest.py`
- Test: `tests/test_db.py`, `tests/test_teams.py`

**Interfaces:**
- Produces:
  - `config.get_settings() -> Settings` (cached; fields per Global Constraints, plus `odds_api_key: str | None`)
  - `db.connect(path: str | Path | None = None) -> sqlite3.Connection` (row_factory `sqlite3.Row`, `PRAGMA foreign_keys=ON`, creates parent dir)
  - `db.init_schema(conn) -> None` (idempotent, `CREATE TABLE IF NOT EXISTS` for every table above)
  - `db.upsert(conn, table: str, rows: list[dict], key: list[str]) -> int` (`INSERT ... ON CONFLICT(key) DO UPDATE`; returns rows written)
  - `db.read_df(conn, sql: str, params=()) -> pd.DataFrame`
  - `teams.TEAMS: list[Team]`, `Team` frozen dataclass (`nba_id:int, abbr:str, name:str, odds_name:str, espn_abbr:str, conference:'East'|'West', division:str`)
  - `teams.by_nba_id(i)`, `teams.by_abbr(s)`, `teams.by_odds_name(s)`, `teams.by_espn_abbr(s)` → `Team | None`
  - `tests/conftest.py`: fixture `conn` (fresh in-memory-like temp-file DB with schema) — later tasks extend conftest with `fixture_db`.

- [ ] **Step 1: Scaffold.** `uv init --package --name nba_predict`, `uv python pin 3.12`, `uv add nba_api requests pandas numpy scipy scikit-learn pydantic pydantic-settings typer streamlit plotly`, `uv add --dev pytest responses`. `.gitignore`: `.venv/ data/*.db .env __pycache__/ .pytest_cache/`. `.env.example`: `ODDS_API_KEY=`.
- [ ] **Step 2: Write failing tests**

```python
# tests/test_db.py
def test_schema_idempotent(conn):
    init_schema(conn); init_schema(conn)
    names = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert {"games","team_games","odds_snapshots","api_usage","predictions","value_flags",
            "filter_presets","custom_features","backtest_results"} <= names

def test_upsert_updates_not_duplicates(conn):
    row = dict(game_id="g1", season="2026-27", game_date="2026-11-01", home_team_id=1, away_team_id=2, status="scheduled")
    upsert(conn, "games", [row], ["game_id"])
    upsert(conn, "games", [{**row, "status": "final", "home_pts": 110, "away_pts": 100}], ["game_id"])
    rows = conn.execute("select status, home_pts from games").fetchall()
    assert [tuple(r) for r in rows] == [("final", 110)]

# tests/test_teams.py
def test_thirty_teams_unique():
    assert len(TEAMS) == 30 and len({t.nba_id for t in TEAMS}) == 30
    assert sum(t.conference == "East" for t in TEAMS) == 15

def test_lookups():
    assert by_abbr("BOS").nba_id == 1610612738
    assert by_odds_name("Los Angeles Clippers").abbr == "LAC"
    assert by_espn_abbr("NY").abbr == "NYK"
    assert by_espn_abbr("GS").abbr == "GSW" and by_espn_abbr("UTAH").abbr == "UTA"
    assert by_odds_name("Nowhere Team") is None
```

- [ ] **Step 3: Run** `uv run pytest tests/test_db.py tests/test_teams.py -v` → FAIL (import errors).
- [ ] **Step 4: Implement** config, db, teams. Seed `TEAMS` from `nba_api.stats.static.teams.get_teams()` values written out as a literal list (no runtime dependency); ESPN abbreviations that differ from NBA: `NY, GS, SA, NO, UTAH, WSH`; Odds API names equal NBA full names (`"Los Angeles Clippers"`, `"Los Angeles Lakers"`).
- [ ] **Step 5: Run** tests → PASS.
- [ ] **Step 6: Commit** `feat: scaffold package, settings, sqlite schema, team map`.

---

### Task 2: Odds math

**Files:** Create `src/nba_predict/odds_math.py`; Test `tests/test_odds_math.py`

**Interfaces:**
- Produces:
  - `american_to_prob(odds: int) -> float`
  - `american_to_decimal(odds: int) -> float`
  - `devig_two_way(p_a: float, p_b: float) -> tuple[float, float]` (normalize to sum 1)
  - `unit_return(odds: int, result: Literal['win','loss','push']) -> float` (return of a hypothetical 1-unit flag, for evaluation only)
  - `consensus_novig(pairs: list[tuple[int,int]]) -> tuple[float,float] | None` (median of per-book no-vig probs; `None` if empty)
  - `best_price(prices: list[int]) -> int` (highest decimal payout)

- [ ] **Step 1: Write failing tests**

```python
def test_american_to_prob():
    assert american_to_prob(-110) == pytest.approx(110/210)
    assert american_to_prob(150) == pytest.approx(0.4)
    assert american_to_prob(-200) == pytest.approx(2/3)

def test_devig():
    assert devig_two_way(110/210, 110/210) == pytest.approx((0.5, 0.5))
    a, b = devig_two_way(american_to_prob(-150), american_to_prob(130))
    assert a + b == pytest.approx(1.0) and a > 0.58

def test_unit_return():
    assert unit_return(-110, "win") == pytest.approx(100/110)
    assert unit_return(150, "win") == pytest.approx(1.5)
    assert unit_return(150, "loss") == -1 and unit_return(-110, "push") == 0

def test_consensus_and_best():
    assert consensus_novig([(-110,-110), (-120,100)])[0] == pytest.approx(0.5112, abs=1e-3)
    assert consensus_novig([]) is None
    assert best_price([-115, -105, -110]) == -105 and best_price([120, 135]) == 135
```

- [ ] **Step 2: Run** `uv run pytest tests/test_odds_math.py -v` → FAIL.
- [ ] **Step 3: Implement** the functions above in `odds_math.py`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: odds math (implied prob, devig, unit return)`.

---

### Task 3: NBA stats ingest + CLI skeleton

**Files:** Create `src/nba_predict/data/nba_client.py`, `src/nba_predict/ingest.py`, `src/nba_predict/cli.py`, `tests/fixtures/leaguegamelog_sample.json` (≈6 rows: 3 games, both teams, built by hand in LeagueGameLog column format); Test `tests/test_ingest_nba.py`

**Interfaces:**
- Consumes: `db.upsert`, `teams.by_abbr`, `config.get_settings`
- Produces:
  - `NbaClient(delay_s: float, retries: int = 3)` with `.season_team_logs(season: str) -> pd.DataFrame` (raw LeagueGameLog `player_or_team_abbreviation='T'`) and `.slate(date: datetime.date) -> list[dict]` (each `{game_id, game_date, home_team_id, away_team_id, status}`); retries with exponential backoff (1s, 2s, 4s) on `requests` timeouts/HTTP errors, then raises `IngestError(message)`.
  - `ingest.parse_team_logs(df: pd.DataFrame, season: str) -> tuple[list[dict], list[dict]]` → (games rows, team_games rows). Home team = row whose `MATCHUP` contains `" vs. "`; `game_date` from `GAME_DATE`; game `status='final'`, scores from both rows.
  - `ingest.ingest_season(conn, client, season) -> int` (upserts; returns team_games rows written; re-run is a no-op on counts)
  - `ingest.ingest_slate(conn, client, date) -> int` (upserts `games` with `status='scheduled'`, never downgrades a `final` row)
  - `cli.app` (typer) with `nba init-db`, `nba ingest --season S [--season S2 ...]` (default `current_season`), `nba slate [--date YYYY-MM-DD]`

- [ ] **Step 1: Write failing tests**

```python
def test_parse_team_logs(sample_logs):
    games, tgs = parse_team_logs(sample_logs, "2025-26")
    assert len(games) == 3 and len(tgs) == 6
    g = next(x for x in games if x["game_id"] == sample_logs.GAME_ID[0])
    home = next(t for t in tgs if t["game_id"] == g["game_id"] and t["is_home"] == 1)
    assert g["home_team_id"] == home["team_id"] and g["home_pts"] == home["pts"]
    assert all(t["opp_team_id"] != t["team_id"] for t in tgs)

def test_ingest_season_idempotent(conn, fake_client):
    assert ingest_season(conn, fake_client, "2025-26") == 6
    ingest_season(conn, fake_client, "2025-26")
    assert conn.execute("select count(*) from team_games").fetchone()[0] == 6

def test_slate_does_not_downgrade_final(conn, fake_client):
    ingest_season(conn, fake_client, "2025-26")
    fake_client.slate_rows = [{**<an already-final game>, "status": "scheduled"}]
    ingest_slate(conn, fake_client, date(2025, 11, 1))
    assert conn.execute("select status from games where game_id=?", (gid,)).fetchone()[0] == "final"

def test_client_retries_then_raises(monkeypatch):
    # endpoint raises requests.Timeout every call → IngestError after 3 attempts, sleep patched out
```

`fake_client` is a conftest fixture: object with `season_team_logs` returning the fixture DataFrame and `slate` returning `slate_rows`.

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** `season_team_logs` uses `leaguegamelog.LeagueGameLog(season=..., player_or_team_abbreviation="T", timeout=30).get_data_frames()[0]`. `slate(date)`: if `date == today (ET)` use `nba_api.live.nba.endpoints.scoreboard.ScoreBoard()`; else `scoreboardv2.ScoreboardV2(game_date=...)` GameHeader (`GAME_ID, HOME_TEAM_ID, VISITOR_TEAM_ID, GAME_STATUS_ID`; status 3 → `final`). If ScoreboardV2 is unavailable in the installed nba_api, use `scoreboardv3` with the same mapping. Sleep `delay_s` before every call.
- [ ] **Step 4: Run** tests → PASS.
- [ ] **Step 5: Live smoke check** `uv run nba init-db && uv run nba ingest --season 2026-27 && uv run nba slate` → prints row counts; `select count(*) from team_games where season='2026-27'` ≈ 2 × games played so far. Then backfill `--season 2023-24 --season 2024-25 --season 2025-26` → ≈2460 rows each.
- [ ] **Step 6: Commit** `feat: nba_api ingest of team game logs and daily slate`.

---

### Task 4: ESPN past-game lines

**Files:** Create `src/nba_predict/data/espn_client.py`; Modify `ingest.py`, `cli.py`; Create fixtures `tests/fixtures/espn_scoreboard_20261110.json`, `espn_summary_with_odds.json`, `espn_summary_no_odds.json` (save real responses: event 401909960 for with-odds; any 2024-25 event, e.g. 401705127, for no-odds); Test `tests/test_espn.py`

**Interfaces:**
- Produces:
  - `EspnClient(delay_s)` with `.events(date) -> list[dict]` (`{espn_event_id, game_date, home_espn_abbr, away_espn_abbr, completed: bool}`) and `.summary(event_id) -> dict`
  - `espn_client.parse_pickcenter(summary: dict) -> dict | None` → keys `book, home_ml, away_ml, home_spread, home_spread_price, away_spread_price, total, over_price, under_price`; `None` when `pickcenter` missing/empty; individual missing fields → `None`
  - `ingest.ingest_espn_lines(conn, client, date_from, date_to, only_missing=True) -> int`: for each date, match events to `games` by (`game_date`, home team via `by_espn_abbr`), set `games.espn_event_id`, insert one `odds_snapshots` row `source='espn', is_closing=1, pulled_at=<game_date>T23:59:59Z`. With `only_missing`, skip games that already have an espn snapshot. Games without lines get `espn_event_id` set but no snapshot.
  - CLI `nba espn --from YYYY-MM-DD --to YYYY-MM-DD` (default: yesterday→yesterday)

- [ ] **Step 1: Write failing tests**

```python
def test_parse_pickcenter_with_odds(summary_with_odds):
    p = parse_pickcenter(summary_with_odds)
    assert p["book"] == "Draft Kings"
    assert p["home_spread"] == -14.5 and p["total"] == 221.5
    assert p["home_ml"] == -1040 and p["away_ml"] == 670
    assert p["home_spread_price"] == -114 and p["away_spread_price"] == -106

def test_summary_without_pickcenter(summary_no_odds):
    assert parse_pickcenter(summary_no_odds) is None

def test_ingest_matches_and_is_resumable(conn, fake_espn, seeded_games):
    n1 = ingest_espn_lines(conn, fake_espn, date(2026,11,10), date(2026,11,10))
    n2 = ingest_espn_lines(conn, fake_espn, date(2026,11,10), date(2026,11,10))
    assert n1 >= 1 and n2 == 0 and fake_espn.summary_calls == n1   # second run fetched nothing
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Live smoke** `uv run nba espn --from 2026-10-21 --to 2026-11-14` (current season) then 2025-26 regular season (`2025-10-21`→`2026-04-12`, ~25 min at 1 req/s; resumable). Check: `select count(*) from odds_snapshots where source='espn'` ≈ games played.
- [ ] **Step 6: Commit** `feat: ESPN past-game lines ingest`.

---

### Task 5: Enrichment + metrics registry

**Files:** Create `src/nba_predict/analysis/metrics.py`, `src/nba_predict/analysis/enrich.py`; Modify `tests/conftest.py` (add `fixture_db`); Test `tests/test_enrich.py`, `tests/test_metrics.py`

`fixture_db` (conftest): deterministic synthetic league — 4 teams (BOS, NYK, LAL, DEN real ids), 2 seasons × 24 games, every team pair home and away, hand-set box scores, ESPN snapshots on season 2 only, plus 2 scheduled games on the last date + 1. Built with plain `upsert` so later tasks reuse it; hand-checked numbers in tests refer to its first 3 BOS games (document them in the conftest docstring).

**Interfaces:**
- Produces:
  - `metrics.add_game_metrics(df) -> df` adds per-row: `poss = fga + 0.44*fta - oreb + tov`, `game_poss = (poss + opp_poss)/2`, `ortg = 100*pts/game_poss`, `drtg = 100*opp_pts/game_poss`, `net = ortg - drtg`, `pace = 48*game_poss/(min/5)`, `efg = (fgm + 0.5*fg3m)/fga`, `tov_pct = tov/poss`, `orb_pct = oreb/(oreb + opp_dreb)`, `ftr = fta/fga`, `margin = pts - opp_pts`, `win = wl=='W'`
  - `metrics.METRICS: dict[str, Metric]` — `Metric(key, label, kind: 'rating'|'pct'|'mean'|'record', fn: Callable[[pd.DataFrame], float|str])`. Ratings are **sum-based** (`ortg = 100*Σpts/Σgame_poss`), not mean of game ratios. Keys: `ortg, drtg, net, pace, efg, tov_pct, orb_pct, ftr, margin, pts, opp_pts, win_pct, record, ats_record, ats_pct, ou_record, over_pct, cover_margin, total_pts`.
  - `metrics.aggregate(df, metric_keys: list[str], group_by: list[str] | None = None, raw_aggs: dict[str, str] | None = None) -> pd.DataFrame` (always includes `gp`; `raw_aggs` = any numeric column → `mean|median|sum`)
  - `enrich.build_enriched(conn) -> pd.DataFrame` and writes `team_games_enriched`. One row per final team-game: all `team_games` columns + opponent box (`opp_*`), `team_abbr, opp_abbr, conference, division, opp_conference`, game metrics above, schedule (`rest_days` = days since previous game − 1, capped at 7, first game of season = 7; `b2b = rest_days == 0`; `three_in_four`), odds from closing snapshots (`is_closing=1`): the median across `oddsapi` books of `home_spread`/`total`/ML if any, else the `espn` row: `team_spread`, `total_line`, `team_ml`, `is_favorite`, `cover_margin = margin + team_spread`, `ats` ∈ {`cover`,`no_cover`,`push`,NULL}, `ou` ∈ {`over`,`under`,`push`,NULL}, and opponent strength as of the game: `opp_win_pct_pre`, `opp_net_pre`, `opp_net_rank_pre` (season-to-date, before this game; NULL when opponent has < 5 games), `month`, `weekday`.

- [ ] **Step 1: Write failing tests** — `test_poss_and_ratings_hand_checked` (exact values for BOS game 1 from conftest docstring), `test_sum_based_rating` (2-game df: aggregate `ortg` equals `100*Σpts/Σposs`, not mean of ratios), `test_rest_and_b2b` (fixture BOS game on consecutive days → `rest_days==0`, `b2b`), `test_ats_from_home_spread` (home spread −5, home wins by 5 → home `push`, away `push`; wins by 7 → home `cover`, away `no_cover`), `test_opp_strength_pre_game_only` (changing the opponent's result in this same game does not change `opp_win_pct_pre`), `test_missing_odds_null` (season-1 rows: `ats` is NULL, metrics still present).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: enriched team-game table and metric registry`.

---

### Task 6: FilterSpec, presets

**Files:** Create `src/nba_predict/analysis/filters.py`; Test `tests/test_filters.py`

**Interfaces:**
- Consumes: `team_games_enriched` columns (Task 5)
- Produces:
  - `FilterSpec(BaseModel)`: `seasons: list[str]|None`, `date_from/date_to: date|None`, `as_of: date|None` (strictly before), `teams/opponents: list[str]|None` (abbr), `venue: Literal['all','home','away']='all'`, `last_n: int|None (ge=1)`, `rest_min/rest_max: int|None`, `b2b: bool|None`, `role: Literal['any','favorite','underdog']='any'`, `spread_min/spread_max`, `total_min/total_max: float|None`, `result: Literal['any','win','loss']='any'`, `ats: Literal['any','cover','no_cover','push']='any'`, `ou: Literal['any','over','under','push']='any'`, `opp_min_win_pct: float|None`, `opp_net_rank_max: int|None`, `conferences/opp_conferences/divisions: list[str]|None`, `months/weekdays: list[int]|None`
  - `apply_filter(df, spec) -> pd.DataFrame` — all row filters first, **then** `last_n` per team (most recent N by `game_date` of the remaining rows). Odds-based filters exclude rows with NULL odds. Opponent-strength filters exclude rows with NULL `opp_*_pre`.
  - `spec.describe() -> str` (human-readable one-liner for page headers)
  - `save_preset(conn, name, spec)`, `load_preset(conn, name) -> FilterSpec`, `list_presets(conn) -> list[str]`, `delete_preset(conn, name)`
  - CLI `nba explore [--preset NAME] [--team BOS] [--venue away] [--last-n 7] --metrics net,ats_record [--group-by team_abbr] [--csv out.csv]` (reads `team_games_enriched`, prints the `aggregate` table)

FilterSpec is applied in pandas only (no SQL generation) — my call; the SQL console covers hand-written SQL.

Opponent strength uses the opponent's record **as of that game**, not today's — my call (no leakage; consistent with the models).

- [ ] **Step 1: Write failing tests**

```python
def test_last_n_applies_after_other_filters(enriched):
    spec = FilterSpec(teams=["BOS"], venue="away", last_n=3)
    out = apply_filter(enriched, spec)
    assert len(out) == 3 and (out.is_home == 0).all()
    expected = enriched[(enriched.team_abbr=="BOS")&(enriched.is_home==0)].nlargest(3, "game_date")
    assert set(out.game_id) == set(expected.game_id)

def test_each_filter_field(enriched):   # parametrized: one case per FilterSpec field, asserts the
    ...                                 # predicate holds on every returned row and count matches pandas

def test_empty_result(enriched):
    out = apply_filter(enriched, FilterSpec(teams=["BOS"], opponents=["BOS"]))
    assert out.empty and list(out.columns) == list(enriched.columns)
    assert aggregate(out, ["net","record"]).iloc[0]["gp"] == 0

def test_preset_roundtrip(conn):
    s = FilterSpec(teams=["NYK"], last_n=7, opp_min_win_pct=0.5)
    save_preset(conn, "nyk-l7", s)
    assert load_preset(conn, "nyk-l7") == s and "nyk-l7" in list_presets(conn)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: FilterSpec engine and presets`.

---

### Task 7: Feature table (windows, splits, SOS, schedule)

**Files:** Create `src/nba_predict/features/build.py`; Test `tests/test_features.py`

**Interfaces:**
- Consumes: `team_games_enriched`, `games` (incl. scheduled), `apply_filter`, `METRICS`
- Produces:
  - `build_feature_table(conn, windows=(5,8,10,15)) -> pd.DataFrame`, writes `game_features`. One row per game (final **and** scheduled). Column naming `{side}_{scope}_{win}_{metric}`: `side ∈ {h,a}`, `scope ∈ {all, split}` (split = home team's home games / away team's road games), `win ∈ {l5,l8,l10,l15,szn}`, `metric ∈ {ortg, drtg, net, pace, efg, tov_pct, orb_pct, ftr, margin, win_pct, ats_pct, over_pct, cover_margin, sos}`; plus `{side}_{scope}_{win}_gp`, `{side}_rest`, `{side}_b2b`, `{side}_3in4`, `{side}_gp_szn`, diffs `d_{scope}_{win}_{metric} = h − a`, market cols `m_home_spread, m_total, m_home_ml, m_away_ml, m_home_novig` (closing snapshot, NULL if none), targets `margin, total, home_win` (NULL for scheduled), and keys `game_id, season, game_date, home_team_id, away_team_id`. `sos` = mean `opp_net_pre` over the window's games; `{scope}_{win}_adjnet = net + sos` also included.
  - Windows never cross seasons except `l*` windows in the first 15 games of a season, which **do** reach into last season — my call so mid-November L15 is a real 15 games. `szn` is season-only.
  - Rolling ratings are sum-based over the window (rolling sums of `pts`, `opp_pts`, `game_poss`, then ratio), computed with `groupby(...).shift(1).rolling(n, min_periods=1)` so the current game is excluded. Scheduled games get stub rows appended before rolling.
  - `team_profile(enriched, team_abbr, as_of: date, n: int|None, venue: 'all'|'home'|'away', spec: FilterSpec|None = None) -> dict[str, float]` (`n=None` = season-to-date; same metric keys + `gp`): slow per-call path for the playground, honoring an extra `FilterSpec` on input games.

- [ ] **Step 1: Write failing tests**

```python
def test_no_leakage(fixture_db):
    base = build_feature_table(fixture_db).set_index("game_id")
    d = <date of a mid-season game>
    fixture_db.execute("update team_games set pts = pts + 40 where game_date >= ?", (d,))
    <rebuild enriched>; after = build_feature_table(fixture_db).set_index("game_id")
    cols = [c for c in base.columns if c.startswith(("h_","a_","d_"))]
    games_on_or_before = base.index[base.game_date <= d]
    pd.testing.assert_frame_equal(base.loc[games_on_or_before, cols], after.loc[games_on_or_before, cols])

def test_window_matches_team_profile(fixture_db):
    # for 3 sampled games and n in (5, 10, None), venue all/split:
    # features h_all_l5_net == team_profile(..., n=5, venue="all")["net"] (approx 1e-9)

def test_short_history_uses_available_games(fixture_db):
    row = <a team's 3rd game of season 1>
    assert row.h_all_l15_gp == 2 and np.isfinite(row.h_all_l15_net)

def test_scheduled_games_have_features(fixture_db):
    sched = build_feature_table(fixture_db).query("home_win.isna()")
    assert len(sched) == 2 and sched.h_all_l5_net.notna().all()

def test_any_window(fixture_db):
    assert "h_all_l3_net" in build_feature_table(fixture_db, windows=(3,)).columns

def test_profile_honors_filter(enriched):
    p = team_profile(enriched, "BOS", as_of, 10, "all", FilterSpec(opp_min_win_pct=0.5))
    # gp equals count of BOS pre-as_of games vs >=.500 opponents, capped at 10
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Live check** `uv run python -c "from nba_predict...; build_feature_table(connect())"` on the real DB completes in < 60 s; row count = `select count(*) from games`.
- [ ] **Step 6: Commit** `feat: leakage-safe game feature table and team profiles`.

---

### Task 8: Elo

**Files:** Create `src/nba_predict/models/elo.py`; Modify `features/build.py` (merge `h_elo, a_elo, elo_prob` into `game_features`); Test `tests/test_elo.py`

**Interfaces:**
- Produces: `EloParams(k=20.0, home_adv=100.0, revert=0.25, mean=1505.0)`; `run_elo(games: pd.DataFrame, params=EloParams()) -> pd.DataFrame` with `game_id, h_elo, a_elo, elo_prob, elo_margin` (pre-game values; scheduled games get current ratings). `elo_prob = 1/(1+10**(-(h−a+home_adv)/400))`; `elo_margin = (h−a+home_adv)/28`. Update: `k * mov_mult * (result − prob)`, `mov_mult = ((|mov|+3)**0.8)/(7.5 + 0.006*elo_diff_winner)` where `elo_diff_winner` includes home advantage from the winner's side. New season: `r = mean + (1−revert)*(r − mean)`.

- [ ] **Step 1: Tests** — `test_first_game_prob` (both 1505 → `elo_prob == approx(0.6400649)`), `test_zero_sum_update` (sum of ratings unchanged after a game), `test_season_revert` (rating 1605 → 1580 at season boundary), `test_pre_game_values` (game 1 `h_elo == 1505`, no use of its own result).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: Elo ratings feature`.

---

### Task 9: Window rating model

**Files:** Create `src/nba_predict/models/window_rating.py`; Test `tests/test_window_rating.py`

**Interfaces:**
- Consumes: `game_features` columns, `team_profile`
- Produces:
  - `WindowParams(weights: dict[str,float] = {"l5":0.15,"l10":0.25,"l15":0.25,"szn":0.35}, split_weight=0.3, use_sos=True, hca=2.5, b2b_pts=1.5, rest_pts=0.4, rest_cap=3, sigma=12.5, sigma_total=18.0)` (pydantic; weights normalized to sum 1; any window key allowed)
  - `team_rating(prefix_cols: Mapping, params) -> tuple[net, ortg, drtg, pace]`: per window, blend `all` and `split` as `(1−split_weight)*all + split_weight*split` (split NaN → use all); `net` uses `adjnet` when `use_sos`; weighted sum over windows (missing window → renormalize remaining weights).
  - `predict_rows(features: pd.DataFrame, params) -> pd.DataFrame` → `game_id, home_win_prob, pred_margin, pred_total`. `pace_exp = mean(h_pace, a_pace)`; `pred_margin = (h_net − a_net)*pace_exp/100 + hca − b2b_pts*(h_b2b − a_b2b) + rest_pts*clip(h_rest − a_rest, −rest_cap, rest_cap)`; `home_win_prob = Φ(pred_margin/sigma)`; `pred_total = pace_exp/100 * ((h_ortg + a_drtg)/2 + (a_ortg + h_drtg)/2)`.
  - `predict_profiles(h: dict, a: dict, h_sched: dict, a_sched: dict, params) -> dict` — same math from `team_profile` dicts (playground).
  - `fit(train: pd.DataFrame, base=WindowParams()) -> WindowParams`: `scipy.optimize.minimize` on margin MSE over `hca, b2b_pts, rest_pts, split_weight` and window weights (softmax-parameterized); then `sigma = std(margin − pred_margin)`, `sigma_total = std(total − pred_total)`.

- [ ] **Step 1: Tests** — `test_even_teams_hca_only` (identical h/a cols, no rest diff → `pred_margin == 2.5`, prob `== Φ(2.5/12.5)`), `test_b2b_penalty` (home b2b → margin drops by 1.5), `test_rows_match_profiles` (same inputs through both paths agree), `test_predict_with_two_prior_games` (row with only l5/szn finite, others NaN → finite prediction), `test_fit_recovers_hca` (synthetic rows where margin = net diff + 3.0 + noise → fitted `hca ≈ 3.0 ± 0.3`).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: window rating model with fit`.

---

### Task 10: Read-only SQL + custom features

**Files:** Modify `src/nba_predict/db.py`; Create `src/nba_predict/analysis/custom.py`; Test `tests/test_readonly.py`, `tests/test_custom.py`

**Interfaces:**
- Produces:
  - `db.readonly_query(db_path, sql: str, params=(), max_rows=50_000) -> pd.DataFrame`: opens `file:{path}?mode=ro` with `uri=True`, installs `conn.set_authorizer` allowing only `SQLITE_SELECT, SQLITE_READ, SQLITE_FUNCTION, SQLITE_RECURSIVE`; rejects multiple statements; raises `QueryError(msg)` on anything else.
  - `custom.add_feature(conn, name, expression, use_in_model=False)`: `name` must match `^[a-z][a-z0-9_]{0,40}$` and not collide with a `game_features` column; validates by `readonly_query("SELECT game_id, ({expr}) AS v FROM game_features LIMIT 5")`.
  - `custom.list_features(conn) -> pd.DataFrame`, `custom.delete_feature(conn, name)`, `custom.set_use_in_model(conn, name, bool)`
  - `custom.evaluate(db_path, names: list[str] | None = None) -> pd.DataFrame` (`game_id` + one column per feature), `custom.with_custom(features_df, db_path, only_model=False) -> pd.DataFrame` (left-join onto features).

- [ ] **Step 1: Tests** — `test_select_ok`, `test_rejects_writes` (parametrize `DELETE FROM games`, `DROP TABLE games`, `INSERT ...`, `ATTACH ...`, `PRAGMA writable_schema=1`, `select 1; delete from games` → `QueryError`, and `games` row count unchanged), `test_custom_feature_roundtrip` (`add_feature(conn, "l5_split_gap", "h_split_l5_net - a_split_l10_net")` then `evaluate` value equals the pandas expression), `test_bad_expression_rejected` (`"nonexistent_col + 1"` and `"1); delete from games; --"` → `QueryError`, nothing stored).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: read-only SQL and custom features`.

---

### Task 11: ML model + walk-forward backtest

**Files:** Create `src/nba_predict/models/ml.py`, `src/nba_predict/backtest.py`, `src/nba_predict/flags.py` (market functions only; Task 13 adds the rest); Modify `cli.py`; Test `tests/test_ml.py`, `tests/test_backtest.py`, `tests/test_markets.py`

**Interfaces:**
- Consumes: `game_features`, `custom.with_custom`, `run_elo`, `window_rating.fit/predict_rows`, `odds_math`
- Produces:
  - `ml.default_feature_cols(df) -> list[str]`: all `d_*` columns + `elo_prob`, `h_rest, a_rest, h_b2b, a_b2b` + custom features with `use_in_model=1`. Market columns (`m_*`) are **never** inputs.
  - `MLModel(feature_cols, C=1.0, alpha=1.0)`: `.fit(df)` trains three sklearn `Pipeline(SimpleImputer(median) → StandardScaler → est)`: `LogisticRegression(C)` on `home_win`, `Ridge(alpha)` on `margin`, `Ridge` on `total`; `.predict(df) -> DataFrame(game_id, home_win_prob, pred_margin, pred_total)`.
  - Training/scoring rows exclude games where either team has `gp_szn < 3` **and** no prior-season data — prediction still works for them.
  - `backtest.run(conn, test_seasons=("2024-25","2025-26"), refit="monthly") -> pd.DataFrame` (long: `model, season, metric, value`) and writes `backtest_results` with one `run_at`. Walk-forward: for each test month, fit on all games before the month's first day. Models: `home_always` (p = training home win rate, margin = training mean margin), `elo`, `window`, `ml`. Metrics: `n, accuracy, brier, log_loss, margin_mae, total_mae, ece` (10 equal-width bins). For seasons with `m_home_novig`: `market_log_loss`, and for each model, flagged-pick simulation at the spec rule (`evaluate_markets` on `market_view(closing_only=True)`, edge ≥ `edge_threshold`, one side per market, flat 1 unit at the closing price): `bets, hit_rate, roi`.
  - `flags.market_view(conn, game_id, closing_only=False) -> dict | None` → `{ml_novig: (home, away), ml_price: (home, away), ml_book: (home, away), home_spread, spread_novig: (home, away), spread_price: (home, away), total, total_novig: (over, under), total_price: (over, under)}`; source = latest `oddsapi` pull (closing pull when `closing_only`), else ESPN; consensus no-vig via `odds_math.consensus_novig`, best price via `best_price`, lines = median across books; any market without both prices → its keys `None`; `None` if no snapshot at all.
  - `flags.evaluate_markets(pred: dict, market: dict, sigma: float, sigma_total: float) -> list[dict]` (each `{market, side, line, price, book, model_prob, market_prob, edge}`): ML `model_prob = home_win_prob` (away = 1 − p); spread `P(home covers) = Φ((pred_margin + home_spread)/sigma)`; total `P(over) = Φ((pred_total − total)/sigma_total)`; `edge = model_prob − market_prob`; markets with `None` line/novig keys are skipped; `*_price`/`*_book` keys are optional (output `price`/`book` = `None`).
  - When any custom feature has `use_in_model=1`, the backtest also runs `ml_base` (ML without custom features) so the Backtest page shows whether they helped.
  - `backtest.best_model(conn) -> str` = lowest `log_loss` averaged over seasons in the latest run; `"window"` if no run exists.
  - CLI `nba features` (rebuild enriched + features), `nba backtest [--season S ...]` prints a pivot table.

- [ ] **Step 1: Tests** — `test_ml_never_uses_market_cols`, `test_ml_handles_nan` (features with NaN windows → finite probs), `test_backtest_walk_forward_no_future` (spy on `MLModel.fit`: max `game_date` of every training set < min `game_date` of its scoring month), `test_backtest_metrics_shape` (fixture_db → rows for all 4 models × metrics; `home_always` accuracy equals fixture home win share), `test_best_model_default`, `test_ml_base_added_with_custom_feature`, plus `test_evaluate_markets_values` (below) and `test_market_view_missing_prices` (snapshot with spread but no spread prices → spread keys `None`, ML still present).

```python
def test_evaluate_markets_values():
    out = evaluate_markets(dict(home_win_prob=0.6, pred_margin=4.0, pred_total=230.0),
                           dict(ml_novig=(0.55,0.45), home_spread=-3.0, spread_novig=(0.5,0.5),
                                total=226.0, total_novig=(0.5,0.5)), sigma=12.5, sigma_total=18.0)
    ml_home = next(o for o in out if o["market"]=="ml" and o["side"]=="home")
    assert ml_home["edge"] == pytest.approx(0.05)
    sp_home = next(o for o in out if o["market"]=="spread" and o["side"]=="home")
    assert sp_home["model_prob"] == pytest.approx(norm.cdf(1/12.5))
    over = next(o for o in out if o["side"]=="over")
    assert over["model_prob"] == pytest.approx(norm.cdf(4/18))
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Live check** `uv run nba features && uv run nba backtest` on the real DB: every model's `log_loss` < `home_always`'s; report the table to the user verbatim (whatever it shows).
- [ ] **Step 6: Commit** `feat: ML model and walk-forward backtest`.

---

### Task 12: Odds API live odds

**Files:** Create `src/nba_predict/data/odds_client.py`; Modify `ingest.py`, `cli.py`; Create `tests/fixtures/oddsapi_nba.json` (2 events, 2 books, one event with an unknown team name, one tip at `03:30:00Z`); Test `tests/test_odds_api.py`

**Interfaces:**
- Produces:
  - `OddsApiClient(api_key: str | None)`; `.nba_odds() -> tuple[list[dict], dict]` (events JSON, `{"used": int, "remaining": int}` from headers `x-requests-used`/`x-requests-remaining`/`x-requests-last`). Missing key → `OddsApiError("ODDS_API_KEY not set — add it to .env")`; 401 → `OddsApiError("invalid key")`; 429 or `remaining == 0` → `OddsApiError("quota exhausted")`.
  - `odds_client.parse_events(events) -> list[dict]` one per (event, book) in `odds_snapshots` shape plus `home_name, away_name, commence_time`. Spread: `home_spread` = the home outcome's `point`.
  - `ingest.ingest_odds(conn, client, closing=False) -> int`: maps names via `by_odds_name`, matches `games` by **Eastern** date of `commence_time` + home team; unmatched/unmapped → `logging.warning` and skip; inserts snapshots `source='oddsapi'`, `is_closing=int(closing)`, shared `pulled_at`; writes `api_usage` row (`credits_used = x-requests-last`).
  - CLI `nba odds [--closing]` prints games matched and credits remaining.

- [ ] **Step 1: Tests** (mock with `responses`) — `test_request_params` (asserts exact URL and query `regions=us&markets=h2h,spreads,totals&oddsFormat=american`), `test_parse_events` (book count, spread sign, totals), `test_late_tip_matches_eastern_date` (`2026-11-16T03:30:00Z` → game on `2026-11-15`), `test_unmapped_team_skipped_with_warning` (caplog), `test_quota_exhausted_raises_clean_error` (429 and remaining=0), `test_usage_logged` (credits_used == 3 from header).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Live check (user's key in `.env`)** `uv run nba slate && uv run nba odds` → matches every game on today's slate; `api_usage` last row `credits_used == 3`.
- [ ] **Step 6: Commit** `feat: Odds API live odds ingest`.

---

### Task 13: Predict, value flags, grading, daily

**Files:** Create `src/nba_predict/predict.py`; Modify `src/nba_predict/flags.py`, `cli.py`; Test `tests/test_flags.py`, `tests/test_daily.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `predict.predict_date(conn, date, models=("elo","window","ml")) -> pd.DataFrame`: fits each model on all final games before `date` (window params via `fit`, ML via `MLModel.fit`), predicts that date's games, upserts `predictions` (`params_json` = fitted params / sigma / feature list).
  - Uses `flags.market_view` / `flags.evaluate_markets` from Task 11.
  - `flags.flag_values(conn, date, model: str | None = None) -> pd.DataFrame`: model defaults to `backtest.best_model`; for each side with `edge >= edge_threshold`, upsert into `value_flags`. At most one side per market per game (higher edge). No stake is computed or stored.
  - `flags.grade(conn, date=None) -> int`: settles flags whose game is `final` (skips others): result from scores (spread: `margin + home_spread` vs 0 from the side; total vs line; push on equality), `units = unit_return(price, result)`, closing from latest `is_closing=1` oddsapi snapshot else ESPN: `clv` = for ML `closing_novig_prob(side) − american_to_prob(price)`; for spread/total, line points in the side's favor (`taken − closing` for home spread, etc.).
  - CLI: `nba predict [--date]`, `nba flags [--date]`, `nba grade [--date]`, `nba daily` = `ingest current season` → `slate` → `espn` (yesterday) → `odds` (on `OddsApiError`: print message, continue) → `features` → `predict` → `flags` (skipped without odds) → print table: `game, model win% (home), market win%, edge, proj spread vs line, proj total vs line, flags (B2B/rest/low gp)`.

- [ ] **Step 1: Tests**

```python
def test_flag_threshold_and_one_side(...)  # edge 0.029 not flagged, 0.031 flagged; never both sides
def test_no_odds_no_flag(...)              # game without snapshots: prediction row exists, no flag
def test_grade_spread_push_units(...)      # home -5 at -110, home wins by 5 → push, units 0
def test_grade_skips_unfinished(...)       # scheduled/postponed game → result stays NULL
def test_clv_ml(...)                       # took +150, closing novig 0.42 → clv == approx(0.02)
def test_daily_without_odds_key(...)       # no key: exit code 0, output contains "ODDS_API_KEY",
                                           # predictions rows written for slate games
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Live check** `uv run nba daily` prints today's slate with edges; next day `uv run nba grade` settles them.
- [ ] **Step 6: Commit** `feat: predictions, value flags, grading and daily command`.

---

### Task 14: Dashboard — Today (+ playground), Team trends, Flag log, Backtest

**Files:** Create `app/dashboard.py`, `app/components/data.py`, `app/components/sidebar.py`, `app/pages/1_Today.py`, `app/pages/2_Team_Trends.py`, `app/pages/6_Flag_Log.py`, `app/pages/7_Backtest.py`; Test `tests/test_app.py`

**Interfaces:**
- Consumes: DB tables, `FilterSpec`, presets, `apply_filter`, `aggregate`, `team_profile`, `predict_profiles`, `flags.market_view`, `flags.evaluate_markets`
- Produces:
  - `components.data.load_table(name: str) -> pd.DataFrame` (`st.cache_data`, cache key includes DB file mtime) and `db_path() -> str` (env `NBA_DB_PATH` override — tests point it at the fixture DB).
  - `components.sidebar.filter_sidebar(df, key: str) -> FilterSpec`: one widget per `FilterSpec` field (last-N as a free number input, not a fixed list), preset load/save/delete; shows `spec.describe()`.
  - **Today:** date picker (default today ET); per game card: teams, model win% per model, market no-vig %, edge, best price/book, proj spread vs line, proj total vs line, flags (B2B, rest diff, low gp, "no odds"), and "injuries not included" note. **Playground** expander per game: window multiselect + free "add N" input, weight sliders, split weight, HCA, B2B/rest points, toggle SOS, and a `filter_sidebar` for each team's input games → recomputes with `team_profile` + `predict_profiles` + `evaluate_markets` live; shows the inputs table it used.
  - **Team trends:** team select; line chart of rolling net for each chosen N over the season; table of metrics for L5/8/10/15/szn × all/home/away; ATS and O/U records per window.
  - **Flag log:** filters (date range, market, model); record, hit rate, 1-unit ROI (labelled "model evaluation, not betting"), average CLV; cumulative units chart; calibration chart (model prob deciles vs hit rate); table.
  - **Backtest:** run selector (`run_at`), pivot `model × season × metric`, "Run backtest" button calling `backtest.run`.
  - Charts via plotly; follow the `dataviz` skill when building them.

- [ ] **Step 1: Tests** — `streamlit.testing.v1.AppTest.from_file(page).run()` for each page against `fixture_db` (env `NBA_DB_PATH`): `not at.exception`; Today page shows 2 game cards for the fixture's scheduled date; changing playground HCA slider changes the displayed margin.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Manual check** `uv run streamlit run app/dashboard.py` on the real DB; click through the four pages.
- [ ] **Step 6: Commit** `feat: dashboard today/playground, trends, flag log, backtest`.

---

### Task 15: Dashboard — Explorer, Data, SQL console, Custom features

**Files:** Create `app/pages/3_Explorer.py`, `app/pages/4_Data.py`, `app/pages/5_SQL.py`, `app/pages/8_Custom_Features.py`; Modify `tests/test_app.py`

**Interfaces:**
- Consumes: `filter_sidebar`, `apply_filter`, `METRICS`, `aggregate`, `readonly_query`, `custom.*`
- Produces:
  - **Explorer:** `filter_sidebar` over `team_games_enriched` (+ custom feature columns joined by `game_id`); metric multiselect (registry keys + any numeric column with mean/median/sum); group-by multiselect (`team_abbr, opp_abbr, is_home, rest_days, month, weekday, is_favorite, season`); outputs aggregate table, chart (bar for groups, line over `game_date` when grouped by date), matching game rows, CSV download for both. Empty result → `st.info("No games match these filters")`.
  - **Data:** table picker over every table (`games, team_games, team_games_enriched, game_features, odds_snapshots, predictions, value_flags, api_usage, backtest_results, custom_features`), column chooser, row count, the shared filters where the table has the needed columns, CSV download.
  - **SQL:** text area, Run → `readonly_query`; table + optional chart (x/y column pickers) + CSV; `QueryError` shown with `st.error`; sidebar lists tables and their columns (`PRAGMA table_info` via the read-only path).
  - **Custom features:** list with toggle `use_in_model` and delete; form (name, expression) with "Preview" (first 20 values + distribution) and "Save"; note that ML uses enabled features after the next backtest/predict.

- [ ] **Step 1: Tests** (AppTest on `fixture_db`) — Explorer: set team BOS, venue away, last_n 3 → aggregate `gp == 3` and equals `readonly_query("select count(*) ...")` hand-written equivalent; impossible filter (team BOS, opponent BOS) shows the empty-state info and no exception. SQL: `DELETE FROM games` shows an error and `games` count unchanged. Custom: saving `h_all_l5_net - a_all_l5_net` makes it appear as an Explorer column.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: explorer, data browser, SQL console, custom features pages`.

---

### Task 16: README + end-to-end verification

**Files:** Create/Modify `README.md`

- [ ] **Step 1: Write README:** setup (`uv sync`, copy `.env.example`, get a free key at the-odds-api.com), first-time backfill (`nba init-db`, `nba ingest` ×4 seasons, `nba espn` ranges for 2025-26 and 2026-27 with time estimates), daily routine (`nba daily` morning, `nba odds --closing` ~30 min before first tip, `nba grade` next morning; no bankroll/betting features by design), dashboard (`uv run streamlit run app/dashboard.py`), optional launchd plist / cron line for `nba daily`, data-source caveats (nba_api unofficial, ESPN unofficial and 2023-25 gap, free quota math, no injuries).
- [ ] **Step 2: Full verification** — `uv run pytest -q` all green; run the spec's Verification list on the real DB and record outputs (row counts, backtest table, today's `nba daily` output, credits used = 3).
- [ ] **Step 3: Commit** `docs: README with setup and daily routine`.
