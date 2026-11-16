# Phase 3: Agent C review (Tasks 11–13, ESPN core fallback, and B's phase-3 bugs)

**Scope:** commits `0f1b776..f0b00d2` on `dev`, plus three bugs from `docs/reports/phase-3-B-analysis.md`.

**Suite:** `311 passed` before the review, **`338 passed in 21.09s`** after (`uv run pytest -q`).

**Heavy runs:** every backtest ran on `.backup` copies of `data/nba.db` with `NBA_DB_PATH` pointed at the copy, using `backtest.run(test_seasons=("2024-25","2025-26"), holdout=("2026-27",), n_boot=1000, seed=0)`.

**Writes to the real DB:** exactly one, the one-day ESPN run for game 0022500283, which the coordinator asked for.

## Commits (C)

| commit | message |
|---|---|
| `4de1382` | fix: best_model picks only predicted models; effective HCA counts the Elo prior |
| `4771665` | fix: tune ML regularization per training window, out-of-fold sigma, curated inputs |
| `fbf6a0a` | fix: Elo k/home_adv tuned on 2023-24 (25/40); game_features tracks Elo params |
| `328901d` | fix: record tip times; mark late predictions; never flag a started game (B phase 3) |
| `8ab5bac` | fix: backtest metric 'bets' renamed n_flags (analysis-only wording) |
| `20384fb` | test: phase-3 migrations (start_time_utc, made_after_tip, segment/role) on an old DB |

## Findings

| # | Sev | File:line | Issue | Status |
|---|---|---|---|---|
| 1 | High | `models/ml.py` (`MLModel`) | Point 1. The ML model used about 680 columns with fixed C=1 and alpha=1, and took sigma from in-sample RMS. Its 2026-27 holdout log loss was 0.880, margin MAE 20.4 and ECE 0.21. Totals were fit only on h−a differences, which cannot express pace. | **Fixed.** See the ML section below. |
| 2 | High | `models/elo.py:12` | B's bug. `home_adv=100` put +3.6 pts of home edge into `elo_margin` against about +1.8 actual. Elo over-predicted the home side in every calibration bin. | **Fixed.** k=25 and home_adv=40, tuned on 2023-24 only. See the Elo section below. |
| 3 | High | `flags.py:flag_values`, `predict.py` | B's bug. Predictions were written after tip-off. `status='scheduled'` does not prove a game is pre-tip: the slate maps in-progress games, and games finished but not yet in the log, to `scheduled`, and it can lag. So `flag_values` could flag a game that had started. | **Fixed.** New `games.start_time_utc` comes from the live board `gameTimeUTC`, the V2 status text `7:30 pm ET`, or the Odds API `commence_time`. New `predictions.made_after_tip` is 1/0/NULL. `flag_values` skips games past their tip and predictions marked late. |
| 4 | Med | `backtest.py:best_model` | It could return `home_always`, `window_noprior` or `ml_base`, none of which `predict_date` writes. `nba daily` would then flag nothing, silently. | **Fixed.** The choice is limited to `PREDICTABLE = (elo, window, ml)`. |
| 5 | Med | `backtest.py:effective_hca` | Point 2. With the Elo prior on, it reported about 0.9, which was misleading. The old formula ignored the prior's home advantage and added the full raw hca, even though the prior down-weights hca by (1−w). | **Fixed.** The new definition is `mean(pred − pred_neutral)` over the training data, where neutral means hca=0, split_weight=0 and `elo_margin − home_adv/28`. It is now about 2.0 (2.11 / 1.98 / 1.93), which matches B's 2.1 and the actual +1.8. |
| 6 | Med | `features/build.py:features_fingerprint` | The fingerprint did not include the Elo parameters, so changing Elo defaults would have reused stale `elo_*` columns. | **Fixed.** `EloParams().model_dump()` is now part of the fingerprint. |
| 7 | Low | `flags.py:flag_values` | A re-run could add the opposite side next to a *settled* flag on the same market. That can't happen today, because only final games get settled and only scheduled games get flagged. | **Fixed (defensive).** Markets that already have a settled flag are skipped. |
| 8 | Low | `backtest.py:135` | The metric was named `bets`, which is betting wording (point 8). | **Fixed.** Renamed `n_flags`. A guard test now scans the schema and metric names. |
| 9 | Info | game `0022500283` | Not a code bug. The core fallback only runs for the dates being processed, and the earlier backfill stopped at 2025-11-23. | **Fixed by running it.** `nba espn --from 2025-11-24 --to 2025-11-24` printed `wrote 2 ESPN snapshots`: an ESPN BET close (IND +9.5, ML +320 / −425, total 235.5) and an open. The 2025-26 regular season now has a closing line for 1230/1230 games. |
| 10 | Info | `backtest.py:_boot_ci` | A single shared RNG (seed 0) is consumed in model/segment order. CIs reproduce run to run but shift if the set of models changes, and the market block has no CI. | Open (note). |
| 11 | Info | plan/spec file names | `docs/superpowers/*/…-nba-betting-aid*.md` and the plan's "`bets, hit_rate, roi`" wording live in docs I don't edit. | Open (coordinator). |

## ML model: before and after

All numbers are on a backup copy, walk-forward with monthly refit. "after" is the committed default (`FEATURE_SET="curated"`). "all-tuned" uses the same tuning on every `d_*` column.

**What changed:**
- C and alpha are tuned per training window by forward-chaining whole-date folds (3 folds) over `C_GRID` 0.001–1 and `ALPHA_GRID` 1–3000. Each fit picks its own value.
- `sigma` and `sigma_total` are the RMS of the out-of-fold residuals.
- About 40 curated inputs, plus `t_*` pace and expected-points sums for totals.

| season (role) | metric | before | all-tuned | **after (curated)** |
|---|---|---|---|---|
| 2024-25 (test) | log loss | 0.6734 | 0.6124 | **0.6111** (later 0.6130 with the new Elo) |
| | ECE | 0.0877 | 0.0064 | 0.0152 |
| | margin MAE | 12.24 | 11.15 | **11.14** |
| | total MAE | 18.47 | 14.85 | **14.73** |
| | early-segment log loss | 0.8182 | 0.6243 | 0.6278 |
| 2025-26 (test) | log loss | 0.6731 | 0.6042 | **0.6036** |
| | ECE | 0.0949 | 0.0231 | 0.0298 |
| | margin MAE | 12.52 | 11.52 | **11.48** |
| | mae_gap vs market | +1.55 | +0.53 | **+0.49 [0.30, 0.68]** |
| | total MAE | 18.48 | 15.35 | **15.20** |
| 2026-27 (holdout) | log loss | **0.8797** | 0.6002 | **0.5991** |
| | margin MAE | 20.40 | 11.79 | 11.95 |
| | total MAE | 22.69 | 14.95 | 15.45 |
| | ECE | 0.2092 | 0.0419 | 0.0809 |

**Chosen values:** curated C = 0.003 and alpha ≈ 1000 (interior of the grid). With all columns, C hit the grid floor (0.001) and alpha the ceiling (3000), which means more shrinkage was still wanted.

**Why curated:** it is equal or better on both test seasons, it is 40 columns instead of about 680, and it's faster. ML is now in the same league as the window model (0.6036 vs 0.6023 log loss in 2025-26) instead of far behind.

## Elo `home_adv` / k: before and after

**Tuning:**
- Grid: k ∈ {20, 25, 30, 35, 40} × home_adv ∈ {20, …, 100}, scored by log loss on **2023-24 final games only**.
- Best: k=25, home_adv=40, log loss 0.6205. The surface is flat between 40 and 50 (0.6209 at k=25, home_adv=50).
- On 2024-25 and 2025-26, which were not used for tuning: log loss **0.6191 → 0.6059** and margin bias **+1.85 → −0.35**.
- The window model was refit with the new prior; its fitted `elo_prior_games` is now 20–23 (was 17–18).

| season (role) | Elo log loss | Elo ECE | Elo MAE | window log loss | window MAE | window effective HCA |
|---|---|---|---|---|---|---|
| 2024-25 test | 0.6260 → **0.6113** | 0.076 → 0.032 | 11.26 → 11.18 | 0.6058 → 0.6061 | 11.04 → 11.04 | 0.91 → **2.11** (new definition) |
| 2025-26 test | 0.6122 → **0.6006** | 0.067 → 0.035 | 11.60 → 11.54 | 0.6020 → 0.6023 | 11.46 → 11.47 | 0.89 → **1.98** |
| 2026-27 holdout | 0.6131 → **0.5938** | 0.109 → 0.083 | 11.96 → 11.76 | 0.6060 → **0.5994** | 11.90 → 11.76 | 0.99 → **1.93** |

- Market 2025-26 closing log loss is unchanged at **0.5746**. The best model (window) is at 0.6023, and its mae_gap is **+0.48 [0.30, 0.65]**.
- `best_model` is still `window`, by mean test log loss: 0.6042 vs Elo 0.6060 vs ML 0.6086.
- Elo now beats the window model on 2025-26 and on the holdout, so Elo may become the choice once 2026-27 test rows exist.
- Flagged-side 1-unit ROI is still negative for every model, for example window −0.088 and ML −0.106 in 2025-26. That is an evaluation metric only.

## Verdicts on A's points 1–8

1. **ML is bad: confirmed, and fixed test-first** (#1). Tests:
   - `test_ml_tunes_regularization_on_time_ordered_folds`: shuffled input is re-sorted, folds chain strictly forward, and alpha shrinks noise.
   - `test_ml_sigma_is_out_of_fold_not_in_sample`
   - `test_fixed_params_skip_tuning`
   - `test_total_uses_both_teams_pace_not_only_differences`
   - `test_curated_feature_set`
2. **Effective HCA: yes, the old figure was misleading. Definition fixed** (#5). Tests pin hca plus the split edge when the prior is off, and the prior's `home_adv/28` when the prior dominates.
3. **Backtest: correct.**
   - The `segment` and `role` migration works; it is tested on an old-schema DB.
   - Bootstrap B=1000 with seed 0 resamples games and pairs model and market per game.
   - `mae_gap` = model MAE − |margin + home_spread|, i.e. the market predicts −home_spread. Verified in code and against B's numbers.
   - `--holdout` uses one frozen pre-season fit, is written with `role='holdout'`, and never reaches `best_model`, which filters to `role='test'`.
   - `eligible()` filters training rows only, and every final test game is scored.
   - Gap found and fixed: `best_model` could choose a model that is never predicted (#4).
4. **Odds API: correct.**
   - A 200 response with `remaining=0` returns the data and logs a warning (new test).
   - A 429, or a 401 whose body contains `OUT_OF_USAGE_CREDITS`/quota, gives `quota exhausted`.
   - Any other 401 gives `invalid key`.
   - A missing key makes no call.
5. **flag_values: mostly correct, with one hole closed.**
   - Already true: only `status='scheduled'` games are considered, unsettled flags are deleted before re-flagging, settled flags are never updated, sigma comes from `params_json` (new test computes the exact edge), and only one side per market is flagged.
   - The hole: "scheduled" did not guarantee pre-tip (#3).
   - Defensive fix added for the settled-plus-other-side case (#7).
6. **CLV definitions: correct.**
   - ML: closing no-vig probability of the side − implied probability of the taken price.
   - Spread: taken line − closing line, from the side's perspective; the away line is −home_spread.
   - Total: for the over, close − taken; for the under, taken − close.
   - Spread and total CLV are in points. Existing tests cover both.
7. **ESPN core API: correct.**
   - `espn_core_checked` is set after every core call, including a 404, so each game is checked once. A transient error leaves it at 0, so the game is retried later.
   - Opening lines are stored as `is_closing=0` at `T12:00:00Z`.
   - `source_fingerprint` and `closing_lines` only read `is_closing=1`. A new test shows adding an open row leaves both fingerprints unchanged.
   - `market_view(closing_only=True)` with only an open row returns `None`.
8. **Hard rule: held.**
   - Grepped `src/`, `tests/`, README, `pyproject.toml` and `.env.example`; `app/` doesn't exist yet. There is no bankroll, stake, Kelly, bank, wallet or sportsbook-account code.
   - The only hits are disclaimers and ESPN fixture text such as "ESPN BET" and "GAMBLING PROBLEM?".
   - The `bets` metric was renamed (#8), and a guard test now covers the schema and metric names.

## Tests added (27; 311 → 338)

- **`tests/test_phase3_review.py`** (new, 20 tests):
  - `best_model` only returns predictable models (3 cases)
  - effective HCA with the prior off and on
  - Odds API 200 with 0 credits remaining
  - only scheduled games are flagged
  - settled flags are never touched
  - spread edge uses the `params_json` sigma
  - `closing_only` never picks an open row
  - open rows leave the fingerprints unchanged
  - V2 status text → UTC start time
  - the live board carries the start time
  - the slate stores the start time
  - `made_after_tip` is 1/0, and NULL when the start is unknown
  - a game that has started is never flagged
  - a prediction made after tip is never flagged
  - no betting vocabulary in the schema or metric names
  - old-DB migration
- **`tests/test_ml.py`:** 5 tests (above).
- **`tests/test_elo.py`:**
  - tuned defaults
  - the features fingerprint tracks Elo parameters
  - the formula tests now pin the plan's k=20/100 explicitly
- **Adapted:**
  - `test_backtest.py`: `best_model` among predictable models, and `n_flags`
  - `test_features.py`: `elo_margin` = `home_adv/28`
  - `test_ingest_nba.py`: the live slate now has `start_time_utc`

## Next

- **Bug fix / feature (A, Phase 4 dashboard):** "flag quality" calibration next to every flag. B found the 0.03 edge threshold fires on about 80% of Elo ML markets. Show, per model and market, the backtest hit rate and the mean of (model − market) by edge bucket. Gate flags on a per-model, per-market threshold chosen on 2025-26 validation only, never on the 2026-27 holdout. `effective_hca`, the start/tip times and `made_after_tip` should be shown on the Today page ("prediction made after tip-off: excluded from live record").
- **Feature (data):** run `nba espn --with-open` over 2025-11-24..2026-04-12 to get DraftKings-era opening lines. Then build B's line-movement view on one book instead of the ESPN BET early-season subset.
- **Small:** extend `C_GRID` below 0.001 and `ALPHA_GRID` above 3000 if anyone switches `FEATURE_SET` back to `all`, because those choices sat on the grid edges.
- **Small:** give the market block its own bootstrap CI, and draw one RNG per (model, segment) so adding a model doesn't move other models' CIs.
