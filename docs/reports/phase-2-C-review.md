# Phase 2 — Agent C review (Tasks 5–10, plus A's fix commits)

This covers commits `76aeeca..0cef883` on `dev`, reviewed against the plan (Tasks 5–10, Global Constraints, Review Focus). It also covers B's Phase 1 bugs.

- Start: `198 passed`
- End: **`240 passed in 9.22s`** (`uv run pytest -q`)

## Commits (C)

| commit | message |
|---|---|
| `3a5e7ae` | fix: rebuild team_games_enriched when its source rows changed (point 8, plus a stronger leakage test) |
| `4fa83e3` | fix: harden readonly_query against memory DoS; keep custom expressions inside their parens |
| `237c5e6` | fix: stable sorts before merge_asof so same-date state rows keep game order |
| `a344f46` | fix: ESPN 404 summary no longer aborts nba espn (B's Phase 1 bug) |
| `48172e8` | test: one team's bad MIN never reaches pace in enriched, features or team_profile |

## Findings

| # | Sev | File:line | Issue | Status |
|---|---|---|---|---|
| 1 | High | `features/build.py:158` (old `_load_enriched`), `cli.py:154` | Point 8: an existing `team_games_enriched` was reused unconditionally, so new games, box-score fixes and new closing lines were silently ignored. It also made the new leakage test pass vacuously. | **Fixed.** `enrich.load_enriched` compares a fingerprint stored in a new `derived_meta` table and rebuilds on mismatch. The fingerprint covers final team_games (count, max date, box sums, season types) and closing snapshots (count, max id, line sums, max pulled_at). This is stricter than "row count + max date", because ESPN backfills change odds without changing team_games. |
| 2 | High | `tests/test_features.py:37` | The plan's leakage test added +40 to *both* teams' points, so margin, W/L, net, SOS and ATS didn't change. Flipping `allow_exact_matches` in `enrich._opp_strength` (a same-day leak) still passed it. | **Fixed (tests).** `tests/test_leakage.py` shifts the home side only (flipping results) and deletes later games. It checks that features on or before D are unchanged and that games after D do change, and it pins opponent strength and rest on or before D. Both leaks I planted, in `build._side_features` and in `enrich._opp_strength`, now fail a test. |
| 3 | High | `db.py:250` `readonly_query` | Memory DoS: `randomblob(9e8)`, `zeroblob(5e8)` and `replace(hex(zeroblob(4e6)),…)` each allocated 0.1–1 GB inside the dashboard process. Many large rows were also bounded only by the 30 s timeout. | **Fixed.** `SQLITE_LIMIT_LENGTH` is set to 10 MB per value, and there is a 256 MB result budget (`max_bytes`). Both raise `QueryError`. |
| 4 | Med | `analysis/custom.py:28` | A custom expression could escape its `( … )`. `h_rest) AS v FROM game_features --` passed validation (2 columns), then commented out every later feature in `evaluate()`. `1) AS v FROM games UNION SELECT game_id, (1` changed which rows came back. Reads only, so no writes, but wrong data. | **Fixed.** `check_expression` rejects `;`, `--`, `/*` and any `)` that closes past depth 0, skipping quoted strings and identifiers. Legitimate subqueries, `CASE` and literal parentheses still pass. |
| 5 | Med | `analysis/custom.py:68` | One stale feature, for example one that references `l5` columns after a rebuild with other windows, made `evaluate()` (and so `with_custom`) fail for every feature. | **Fixed.** It falls back to evaluating features one at a time, with a warning per broken feature. |
| 6 | Med | `data/espn_client.py:_get`, `ingest.py:~205` | B's bug: an ESPN 404 summary was retried 3 times and then aborted `nba espn`. Every resume hit the same event again. | **Fixed.** 404/410 raise `EspnNotFound` with no retry, and the game is linked with no snapshot. Other 4xx except 429 fail fast. **Live:** `uv run nba espn --from 2026-04-13 --to 2026-06-13` printed `wrote 52 ESPN snapshots; final games 91, linked 91, with lines 83`. Eight 404 events were logged and skipped. Re-running wrote 0 snapshots. |
| 7 | Low | `features/build.py:51,108`, `analysis/enrich.py:63,71` | Sorts before `merge_asof` were unstable. With two games for one team on one date (a data glitch), the next game could pick up an intermediate state: `gp_szn` came out 6 instead of 7. | **Fixed** with stable sorts. There are no same-date doubles in real data. |
| 8 | Low | `analysis/enrich.py:_schedule` / `features/build.py:_schedule` | With the same glitch, `rest_days` becomes −1, because only the upper end is clipped. | Open (glitch only). |
| 9 | Low | `models/window_rating.py:169` | `sigma = np.std(resid)` subtracts the mean residual. For Φ(margin/σ) the RMS is the right measure. They are equal while the fitted hca leaves about 0 mean residual. | Open. |
| 10 | Low | `features/build.py:116` | For `l<N>` with N > 15, the window has N games while c < 15 and then drops to c games at c = 15. So `gp` isn't monotonic for long windows. It follows the documented rule. | Open (note). |
| 11 | Info | `analysis/custom.py:evaluate` | Custom features read `game_features` as stored, so they are stale until `build_feature_table` runs again. | Open (see Next). |
| 12 | Info | `db.py:_ro_authorizer` | Pinned with tests and verified: writes through CTEs, `ATTACH`, `VACUUM` / `VACUUM INTO` (no file written), temp tables and views, `SAVEPOINT`/`BEGIN`/`ANALYZE`, `pragma journal_mode=`, and `pragma_database_list` / `user_version` / `integrity_check` are all denied, as are a second statement and `load_extension`. Oversized `printf` returns NULL, so nothing is allocated. A runaway recursive CTE stops at the timeout. Allowed and harmless: `PRAGMA table_info(x)`, `sqlite_master`, and a trailing `-- comment`. | OK |

Analysis-only check: `src/` contains no bankroll, stake, Kelly, wager or wallet code. `unit_return` stays an evaluation metric.

## Verdict on A's points 1–8

1. **Cross-season `l<N>` windows: leakage-safe and sensible.**
   - Every state row comes from games strictly before D. The rule picks between two prior-only sums, so it can't leak.
   - Reaching back while there are fewer than 15 prior games *in that scope* means a split l15 reaches back until about the 15th home game. On real 2026-27 data, `h_split_l15_gp` is 15 at the median while `h_split_szn_gp` is 3.
   - That's consistent with "a real 15 games". It does mix in last season's playoffs and roster, which the backtest will reveal.
   - **`team_profile` agrees: 2400/2400 values matched** on the real DB. I sampled 25 games (15 early 2026-27, 10 from 2025-26) × both sides × all/split × l5/l15/szn × 8 metrics, including `gp`, `sos` and `adjnet`.
2. **`merge_asof(allow_exact_matches=False)` on date: correct.**
   - Flipping it to `True` fails 7 tests, including `test_no_leakage`.
   - The original leakage test was weak on margin-type features (#2). The new one catches both planted leaks.
   - Two games for one team on the same date can't see each other, since both get the same pre-date state. The unstable-sort bug (#7) is fixed and covered by a test.
3. **Pace guard `game_min = max(min, opp_min)`: correct, and it covers every pace path.**
   - Per game in `add_game_metrics`, which is stored as `game_min` in enriched.
   - The feature table's rolling sums (`COMP` includes `game_min`).
   - `team_profile`, through `METRICS["pace"]`, which sums `game_min`.
   - Nothing else uses a team's own `min` (grep). The test for this is in `test_features.py`.
   - Live: 0022600231 shows game_min 240 and pace 102.4 for both teams. It's still the only row with `min < 240`, and the rest of its box score is consistent (116 = 2·42 + 4 + 28).
4. **Odds summaries: correct.**
   - Taking the median ML in decimal space avoids the −100/+100 discontinuity, for example −105/+105 → +100.
   - `is_favorite` comes from the spread, falls back to the ML only for 0 or NaN spreads, and stays NA otherwise.
   - Known limit: if Odds API closing rows exist but have no ML, there's no fallback to the ESPN ML.
5. **Opponent strength: correct.**
   - It is season-to-date, strictly before D (pinned in `test_leakage.py`), and NULL under 5 games.
   - Playoffs are included in the record. That's fine for "how good is this opponent", and the rank stays across all 30 teams.
6. **Window model: the fallbacks are reasonable, and the low hca is expected, not a bug.**
   - On the real 2024-26 regular season, the fit gives hca 0.80, split_weight 0.249, weights l15 0.61 / szn 0.33 / l10 0.06 / l5 ≈ 0, b2b 2.56 and σ 14.5.
   - Split ratings already carry home advantage: home-split net is +1.74 above all-games net, and road-split net is −1.77 below it. At split_weight 0.25 that's about 0.87 pts. So the effective HCA is about 0.8 + 0.87 ≈ 1.7.
   - That matches the real mean home margin (+1.68). With split_weight 0 the implied hca is 1.72; with split_weight 1 it is −1.89.
   - Report "effective HCA" in the backtest rather than the raw `hca`.
7. **`readonly_query`: the write and side-effect surface is solid** (#12). I added the DoS guards (#3) and closed the custom-expression escape (#4).
8. **Stale derived tables: fixed** (#1), with tests written first: `test_load_enriched_rebuilds_when_stale` and `…_without_fingerprint`.

## Tests added (42; 198 → 240)

- **`tests/test_leakage.py` (new):**
  - asymmetric change plus removal of later games, which is also not vacuous
  - opponent strength and rest are strictly pre-game
  - two games on one date
- **`tests/test_enrich.py`:**
  - rebuilds on a new game, a box fix and a line change, and reuses when fresh
  - a legacy table without a fingerprint is rebuilt
- **`tests/test_readonly.py`:**
  - 12 side-effect statements
  - `VACUUM INTO` writes no file
  - 3 huge-value DoS cases
  - oversized `printf` returns NULL
  - result-size budget
  - timeout
- **`tests/test_custom.py`:**
  - 6 escape attempts
  - 4 legitimate expressions
  - one broken feature doesn't break the rest
- **`tests/test_espn.py`:**
  - a 404 is not retried
  - a 404 game is linked and skipped on resume
  - 400/403 fail fast while 429/500 are retried
- **`tests/test_features.py`:** a bad team MIN doesn't reach pace in enriched, features or `team_profile`

## Live checks (on a `.backup` copy, then the real DB once B finished)

- `build_feature_table`: 4156 rows = `count(*) from games`, 496 columns, 0.6 s. Enriched has 8294 rows. A reuse load takes 0.06 s.
- Correlation with margin, regular season: 0.407 for `d_all_l10_adjnet` and 0.405 for `d_all_szn_net`. Correlation of `elo_prob` with `home_win`: 0.388.
- `nba explore --team MIL --venue away --last-n 7 --metrics record,net,pts` gives `0-7`, 103.43 pts. A hand-written SQL query gives `0-7 | 103.429`.
- `nba status` after the 404 fix shows 2025-26 playoffs at 85 linked and **77 with lines**, and play-in at 6/6. `fav_disagree` is 1, which is B's MEM@IND case.

## B's numbers

Each query was run from `docs/reports/phase-1-B-analysis.md` against the real DB, read-only.

| B section | result |
|---|---|
| §3 home advantage (win % / mean margin per season) | **Reproduced exactly:** 54.3 / 2.15, 54.4 / 1.63, 55.4 / 1.74, 52.7 / 1.83 |
| §5 2025-26 line coverage by month, first line 2025-11-24 | **Reproduced exactly:** 80/0, 219/48, 197/197, 233/233, 166/166, 239/239, 96/96; first line 2025-11-24 |
| §6 sign check | ML/spread favorites agree in 972/973 and 188/188 games; both MLs negative in 46 and 12 games: **exact**. The 2026-27 cover/over/MAE row is **exact** (48.9 / 46.3 / 11.26 / 17.96 / 0.41 / 14.03). The 2025-26 regular row reproduces to within 0.1 point: I get 50.9 / 48.3 / 18.37 against B's 50.8 / 48.4 / 18.36, with n = 977 and mean residual −0.28, MAE 11.06 and total MAE 14.68 all identical. That looks like a hand-rounding or transcription slip; the conclusion is unchanged. |
| §7 `min < 240` | **Reproduced:** only 0022600231 DEN 168 |

B's two bugs are both handled: the ESPN 404 is fixed (#6), and the pace-from-minutes issue was already covered by A's guard, which I verified (point 3).

## Next

- **Bug fix (A, Phase 3 prep):** custom features and `predict` read `game_features` as stored. Give `build_feature_table` the same fingerprint check, or add a `load_features(conn, windows)` that compares `derived_meta['game_features']` (the enriched fingerprint, plus the games table's count, max date and status counts, plus the windows) and rebuilds on mismatch. Make `custom.evaluate` and `nba daily` call it. Test: add a final game, call `evaluate`, and check the new game's value is there.
- **Feature (data engineering), from B:**
  - Add an ESPN core-API odds fallback (`…/events/{id}/competitions/{id}/odds`, first provider that isn't live). It should fill the 251 games missing in 2025-26 and the 8 playoff games that return 404, stored under their real book (ESPN BET).
  - Store the open line as an `is_closing=0` snapshot.
  - Then add a "line movement" Explorer metric: close − open spread, and the cover rate of the side the line moved toward.
- **Schema (A):** add `games.neutral_site` and exclude those games from the Elo home bonus and from home/road split windows. That affects 11 known games, plus the 2023-24 Vegas and Paris games that the log doesn't flag.
- **Small fixes:**
  - clip `rest_days` at ≥ 0 (#8)
  - use RMS rather than std for `sigma` (#9)
  - report "effective HCA = hca + split_weight × (home split − road split)" in the backtest output, so the about 0.8 raw hca doesn't look like a bug
