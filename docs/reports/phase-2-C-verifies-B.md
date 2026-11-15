# Phase 2 — C verifies B's analysis

**Setup:**
- Source: `docs/reports/phase-2-B-analysis.md` (committed in `fa711e0`).
- DB: a read-only `.backup` copy of `data/nba.db` taken at HEAD `fa711e0`. It holds the same games as B's copy (final games through 2026-11-14, 4147 final plus 9 scheduled).
  - The only data difference from B's copy is the 2025-26 playoff ESPN lines: 77 here vs 25 for B, from my 404 fix.
  - None of B's market sections use playoff lines, and the window model doesn't use odds.
- Code is newer than B's run: A's `06b0d7f` (rest clamp, RMS sigma) and `0f1b776` (game_features fingerprint). Neither changes any number checked below.
- Features were rebuilt on two copies: default windows (4156 × 496) and windows 3,5,8,10,15,20 (4156 × 684). Both shapes match B's.
- I used B's appendix scripts (`q1.py`, `q2.py`, `q4b.py`) and the SQL from §4, unchanged apart from paths. I added paired bootstraps and a binomial test.

**Code changes:** none. I found no bug in the code paths B's numbers depend on.

## Reproduced / not reproduced

| B's claim | My number | Verdict |
|---|---|---|
| §2: L20 adjnet is the best all-games window on 2025-26, MAE **11.57** vs mean-only **13.27** (common sample n=1112) | Common test n **1112**, mean-only **13.27**, `d_all_l20_adjnet` corr **0.432**, MAE **11.57**. The rest of the top-6 matches too: szn net 11.51, szn adjnet 11.60, L20 net 11.62, L15 net and adjnet 11.66. | **Reproduced exactly.** Caveat: the lowest MAE belongs to season-to-date *net* (11.51). L20 adjnet is the best *L-window* and has the best correlation. Neither gap is significant: MAE(L20 adj) − MAE(szn net) = +0.06, 95% CI [−0.12, +0.24]; vs L15 adj −0.09, CI [−0.20, +0.03]. |
| §5: margin MAE, market vs model: **11.06 vs 11.72** (2025-26, n=978); **11.26 vs 12.94** (2026-27, n=188) | **11.06 vs 11.72** (n=978); **11.26 vs 12.94** (n=188). avg(model, market) 11.26 / 11.76, corr 0.88 / 0.80, total MAE 14.68 vs 15.40 and 14.03 vs 14.55. | **Reproduced exactly.** Paired bootstrap of model − market MAE: **+0.65 [+0.42, +0.88]** in 2025-26 and **+1.68 [+0.83, +2.55]** in 2026-27. "The market beats the model" is robust in both seasons. |
| §4: back-to-back costs **1.7–2.3** pts | Team rows: b2b −1.71 vs 1 day of rest +0.25. Home b2b −0.09 vs rested +2.24, so 2.33. Away b2b −3.26 vs −1.52, so 1.74. | **Reproduced.** Controlling for the opponent's b2b at game level (`margin ~ 1 + h_b2b + a_b2b`): home b2b **−2.68**, away b2b **+2.13**. So the effect is about 2.1–2.7, slightly larger than B's range and close to the fitted `b2b_pts` of 2.52. |
| §4: home margin rises about **1 pt per day** of rest advantage, and the market prices it | Bucket table identical: ≤−2 +0.44, −1 −0.12, 0 +1.96, +1 +2.94, ≥+2 +4.56; market +0.45 / +0.33 / +1.85 / +2.87 / +2.94. OLS slope on rest advantage clipped to ±3: **1.21 pts/day** (n=3815). On games with lines: results **0.90** vs market **0.95** pts/day. | **Reproduced.** It is not strictly monotone: the ≤−2 bucket (n=154) sits above −1. The slope fits "about 1/day", and the market's slope matches the results' slope. |
| §5: model gap > 4 pts above the line → home covers **43.0% (n=214)**, mean margin − market −3.05 | **92/214 = 0.430**, resid −3.047. The other buckets match too: 49.5 / 51.6 / 56.5 / 50.5 / 52.9. Split by season: 42.5% (n=153) and 44.3% (n=61). | **Reproduced.** Two-sided binomial p = **0.047** as a single test. With 6 buckets examined it is **not significant** (Bonferroni p ≈ 0.28). Read it as "no evidence the model's disagreement adds information", not "disagreement predicts the opposite". The negative blend weight and `corr(gap, margin − mkt) = −0.05` point the same way. |
| §3: effective HCA ≈ 2.1; refit with no split gives hca **2.18** | Default: hca **1.04**, split_w 0.264, MAE 11.657. No split: hca **2.18**, MAE 11.681. Mean prediction +2.00 / +2.16 vs actual +1.77. Split − all gap: 3.88 / 3.88 / 3.95 / 3.97. Training-set reconstruction: 1.04 + 0.264 × 4.22 × 1.017 = **2.18**. | **Reproduced.** B's 2.09 uses the gap from all seasons (3.9). On the training set the gap is 4.22, which gives exactly 2.18. The 2026-27 refit is the same story (hca 1.06, split_w 0.245). |

All 6 claims reproduce. Every value B printed matches mine to the printed precision.

## Method notes

1. **Out of sample: mostly sound.** Window ranking, the window model and the blend are all fit on earlier seasons and scored on later ones. The 2026-27 test refits on 2023-26. One problem: §2 picks the best of **28 candidate features by their 2025-26 test MAE**, so 2025-26 becomes a selection set, not a clean test set.
   - The L20 recommendation should be confirmed on a season that wasn't used to choose it. On 2026-27 (fit on 2023-25), the ranking holds weakly: all-scope adjnet MAE is L5 12.76, L10 12.46, L15 12.37, **L20 12.31**, szn 12.39 (szn only n=94).
   - The bootstraps above show L20, L15 and season-to-date are statistically tied.
2. **The common-sample filter hides the early season.** Requiring all 28 features to be non-NaN keeps 1112 of 1230 test games and drops the first weeks of the season. That is exactly where window choice and the cross-season reach matter most, so the §2 ranking applies mid and late season only. The early-season weakness appears in §5 (October 2026 MAE 13.74 vs market 11.34), not in §2.
3. **The 2025-26 market comparison starts on 2025-11-24.** ESPN has no lines before that date, so 2025-26 also leaves out the model's worst stretch. The 2025-26 gap (+0.65) is probably an underestimate for a full season.
4. **Leakage: well covered.** B rebuilt the features independently from raw `team_games` and matched `game_features` to 5e-14, and re-derived Elo exactly. Together with my strengthened leakage tests, the feature table is clean. No test-season information reaches any fit; the blend and gap buckets are fit or chosen on 2025-26 and reported on 2026-27.
5. **Multiple comparisons.** B flags the 41.5% overs at |spread| ≤ 3 (−2.6 SE among about 40 cells) as likely chance, which is correct. The same applies to the other notable cells: gap > 4 (above), "both b2b" favourite cover 61.6% (n=73) and 2026-27 overs at 32.4% on |spread| 10+ (n=37). None should drive a feature or a flag without a pre-registered re-test on new data.
6. **Team-perspective rows count each game twice.** The rest and b2b tables in §4 come from `team_games_enriched`: each game appears twice, mirrored, and the home and away rows are correlated. Standard errors computed from n team-rows are too small by about √2. The game-level tables (rest advantage, §5) don't have this problem.
7. **The team-HCA noise argument holds up.** Year-to-year correlations of −0.04, +0.10 and −0.15 are consistent with no persistent team effect, and the noise estimate of about 1.5 matches the observed spread.

## Next

- **Feature (evaluation, A/B): a fixed holdout protocol in `nba backtest`.**
  - Treat 2025-26 as validation, where windows, buckets and thresholds get chosen, and 2026-27 to date as a locked test season that is reported but never tuned on.
  - Report paired-bootstrap CIs for every "model vs market" and "window A vs B" MAE difference.
  - Apply a Bonferroni or Holm adjustment wherever the output ranks more than one bucket or cell.
  - This turns findings like "43% at gap > 4" into a re-test on 2026-27 with a pre-stated hypothesis.
- **Feature (data): report early-season games separately.** Add a `games_into_season` (min of both teams' `gp_szn`) column to `game_features`, and break every backtest and window comparison out by `< 15` vs `≥ 15`. The §2 common sample silently drops the early bucket.
- **Bug-fix-sized (A): drop the 28-feature common-sample requirement from model comparisons.** Compare windows pairwise on their own overlap, or impute adjnet → net, so the early-season games stay in the sample.
