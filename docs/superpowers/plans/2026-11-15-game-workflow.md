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

---

# Addendum: ideas from kyleskom/NBA-Machine-Learning-Sports-Betting (user decisions, 2026-11-15)

**License:** the repo has no license (all rights reserved). Decision: **learn from the ideas, write our own code**. Do NOT copy or paste its source, notebooks, models, data or screenshots, and do not fork it. Codex must not open its code; the feature list below (from its public README) is the whole spec. Describe sources in our README as "inspired by" only.

**Scope the user picked:** XGBoost + neural-net models, a dedicated over/under (totals) model, sportsbook odds from specific books, and layout ideas for the Game Brief. **Excluded:** Kelly, stake sizing, expected-value payout maths, bankroll. Keep model-vs-market probability difference only.

**Training data (user decision):** the **main** win probability on Game Brief stays last-season-only. Longer-history models appear only as a clearly labelled **"Second opinion (long history, 2018-19 onward)"**, never mixed into the lean label, `best_model`, or the flag log. Long history uses seasons already in the DB (2023-24 onward) plus an optional `nba ingest --season` backfill of earlier seasons; do not use seasons before 2018-19.

## Tasks (after the main game-workflow tasks 5-7 are done)
8. **Over/under model.** `models/totals.py`: predicts game total and P(over) from our features (last-season-only, same walk-forward fit as the window/ML models). Feed Game Brief total lean. Add to `nba backtest` (log loss + total MAE + over-hit rate vs closing total).
9. **XGBoost and small neural net.** `models/xgb_model.py`, `models/nn_model.py` (use `xgboost` and scikit-learn `MLPClassifier`; no TensorFlow) for moneyline win prob and totals. Two variants each: `last_season` (main, comparable) and `long_history` (second opinion). Same curated feature set as `models/ml.py`, tuned inside time-ordered folds, out-of-fold calibration (sigmoid). Add to backtest with CIs; report whether they beat `window`/`elo`/market on 2026-27 — if not, say so. Add `xgboost` via `uv add`.
10. **Per-book odds.** `data/book_odds.py`: optional source "sbr" (Sportsbook Review public odds pages) for FanDuel, DraftKings, BetMGM, Caesars, etc., stored in `odds_snapshots` (`source='sbr'`, `book=<name>`). OFF by default (`--source sbr` / `ODDS_SOURCES` setting), 1 req/s, cache, clear message if the page layout changed. Note in README: unofficial source, may break or violate site terms; ESPN and The Odds API remain the defaults. Game Brief shows a per-book line table when present (spread, total, ML per book, best line highlighted) — informational only.
11. **Game Brief polish from their layout idea.** One compact card per game (teams, tip time, each book's ML, model win %, total lean); a sidebar toggle for models shown (window / elo / ml / xgb / nn / second opinion). No Kelly/EV columns.
12. **Report + README.** README section "Models and sources" (what each model is, which are last-season vs long-history, per-book odds caveats). `docs/reports/kyleskom-ideas-codex.md` with what was built, backtest numbers for every new model (verbatim), and `## Next`.

## Acceptance
- No file contains code copied from kyleskom's repo; no Kelly/EV/bankroll/stake wording anywhere (grep).
- New models appear in `nba backtest` with CI; second-opinion models are visibly separate and never affect `best_model` or flags.
- `uv run pytest -q` stays green; no new test files unless a new module has none (then one small file per module is allowed).
