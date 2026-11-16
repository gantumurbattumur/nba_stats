# Game Workflow Redesign Plan (user-directed, 2026-11-15)

**User goal:** the user makes their own prediction; the platform is the *statistics supporter*. Per game: what the market line is, what the model says, how each team has done against the spread / total recently, and a clear "lean" with its evidence. For past games: "predicted X vs line Y, ended Z".

**Hard rules (unchanged):** analysis only. No bankroll, staking, Kelly, bet placement, bank/sportsbook-account features or wording. Caption "Analysis only — no bets placed" on every page. "Lean" = statistical tendency with sample size, never a bet instruction. No new test suites unless noted.

## Decisions (my calls)
1. **Last season only.** Win/loss probabilities and all model fitting use 2025-26 and 2026-27 games only (`model_seasons=("2025-26","2026-27")` in settings). Elo starts at the 2025-26 opener (no earlier seasons). Rolling windows may reach back only into 2025-26. Backtest/validation: fit on 2025-26, test on 2026-27 so far. Today's predictions fit on all games so far in those two seasons. 2023-25 data stays in the DB but is never used by models.
2. **Odds in the UI.** Today's pregame lines come from ESPN's free scoreboard `odds` field (DraftKings spread, total, moneyline) when no ODDS_API_KEY; The Odds API stays the preferred source when a key exists. Store as `odds_snapshots` (source `espn`, `is_closing=0`, `pulled_at` real time). Finished games use the stored closing line.
3. **Simple navigation.** Three main pages: **Game Brief** (home), **Results Review**, **Explore**. All existing technical pages move to an "Advanced" group (Streamlit `st.navigation` sections or numbered `9x_` prefix), unchanged.
4. **Cover lean.** For each game and market (spread, total, moneyline) compute transparent evidence, not a black box:
   - Model margin vs line (points of disagreement) and model total vs line.
   - Each team's ATS record and over/under % in last 5 / 10 / 15 / season, plus the same as home/away and as favourite/underdog, each with n.
   - Lean label: `Strong`, `Lean`, `Slight`, `None`, from a documented rule (agreement count of: model-vs-line direction ≥ threshold, recent ATS trend both teams, venue split). Show each factor as a row with a ✓/✗/– so the user sees why.
   - **Honesty line:** historical hit rate of each label on 2026-27 games with lines (and 2025-26 in-sample, labelled), with n. If n < 30, say "small sample".

## Pages
**Game Brief (home):** date picker (default today ET) → one row per game (teams, tip time, spread, total, ML, model win %). Click/select a game → brief:
1. *Market:* spread/total/ML, source, open vs current if available.
2. *Model:* win % (last-season model), projected margin and total vs line; made-before-tip flag.
3. *Spread:* both teams' ATS last 5/10/15/season (+ home/away + fav/dog splits), lean label + factor rows.
4. *Total:* both teams' over/under % same windows, combined pace trend, lean label + factor rows.
5. *Moneyline:* model win % vs market no-vig %.
6. *Recent games table:* each team's last 10 results with the line, margin, covered?, over/under?.
7. *Head-to-head* last 3 meetings if any (this and last season).

**Results Review:** date range + team filter (reuse FilterSpec). Table per finished game: date, matchup, model pre-game win %/margin/total, closing line, **final score**, covered?, over/under?, "model margin side covered?", "model total side hit?". Summary above: % model-lean-spread covered, % over/under lean hit, calibration by lean label, last-N for a chosen team ("Team A covered 4 of last 5"). Games lacking a pre-tip prediction: reconstruct a *retro* prediction with the same last-season model using only data before that game (labelled "retro") — never mix with real pre-tip predictions without a label.

**Explore:** existing Explorer (filters/metrics/groupby/CSV) as-is, plus preset "ATS & O/U by team".

## Tasks for Codex (in order; one logical change each; user/Claude commits)
1. **Last-season-only models.** Add `model_seasons` setting; restrict Elo, `window_rating.fit`, ML fit, `predict_date`, `backtest` (train 2025-26 → test 2026-27) accordingly. Add an `nba backtest` note listing seasons used. Keep old behaviour behind `--all-seasons` for analysis. Run `uv run pytest -q`; fix tests that encode old defaults.
2. **Pregame odds ingest.** `ingest_espn_scoreboard_odds(conn, date)` reads ESPN scoreboard `competitions[].odds[]` for scheduled games, maps to `games` via team abbr (`teams.by_espn_abbr`), writes `odds_snapshots` (`is_closing=0`, now). Wire into `nba daily` and `nba odds` (used when no key; Odds API when key). Verify with live data for today's date; report fields found.
3. **Trends module** `analysis/trends.py`: `team_cover_trends(enriched, team, as_of, windows=(5,10,15,None), splits)` → ATS %, O/U %, n for all/home/away/fav/dog; `lean(game_row, trends, model_pred, line)` → label + factor list; `lean_history(conn, season)` → hit rate per label with n. Pure pandas, uses only games `< as_of`.
4. **Results module** `analysis/results.py`: build the per-game results table (predicted vs line vs actual), including retro predictions flagged `retro=True`; summaries as above.
5. **UI.** Build Game Brief and Results Review pages; reorganise navigation (Advanced group); plain-language labels, tooltips, sample-size warnings, caption on every page. Keep it uncluttered: no raw column names in main pages.
6. **Smoke check.** AppTest every page against a `.backup` copy of `data/nba.db` (`NBA_DB_PATH`), script outside the repo. Verify: Game Brief shows today's 9 games with lines (or clear "no line yet" when ESPN has none), one game's brief renders all 7 sections, Results Review shows 2026-27 games with covered/over-under columns, no exception. Grep repo for staking/bankroll/bet-placement wording.
7. **Docs.** Update README usage section (new workflow) and write `docs/reports/game-workflow-codex.md` (what changed, commands, results, required `## Next` with a new feature or bug fix). Do not commit reports.

## Acceptance
- Models never touch pre-2025-26 games (assert in a smoke script and report the check).
- A user can open Game Brief, pick a game, and read spread + total + ML evidence in under a minute without SQL.
- Results Review answers "model predicted X vs line Y, final Z; covered? over/under?" and "team A covered N of last 5".
