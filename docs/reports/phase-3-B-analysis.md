# Phase 3 — Agent B data analysis

Run on 2026-11-15, branch `dev`, at HEAD `f0b00d2`. During the run, C committed `4de1382` (`best_model` / effective HCA) and modified `tests/test_ml.py`. I worked only on a fresh copy: `sqlite3 data/nba.db ".backup <scratch>/agentB/nba_p3.db"`. Every script takes that path as `argv[1]`, and the scratch directory as `argv[2]` where it needs one. SQL was run with `sqlite3 -header -column nba_p3.db "<sql>"`. The full scripts are in the appendix, copied verbatim from the scratchpad. A's backtest numbers come from the `backtest_results` rows in the copy (run `2026-11-15T23:59:07Z`, test seasons 2024-25 and 2025-26, segments all/early/rest).

This is analysis only. Cover rates and the "1-unit return" describe how model-vs-market disagreements turned out. They are not staking advice.

## 1. Verifying A's results

| A's claim | Re-derived | Holds? | How |
|---|---|---|---|
| 2025-26 market log loss **0.5746** (n=1306), ESPN closing no-vig | **0.5746**, n=1306 (regular 1223: 0.5694, playoffs 77: 0.6452, play-in 6: 0.733) | Yes | `p3_verify_market.py`: raw `odds_snapshots` (`source='espn' and is_closing=1`), proportional devig, no `nba_predict` code |
| Market margin MAE **11.0256** (−home_spread) | **11.0256**, n=1311 | Yes | same script |
| Window 2025-26: log loss **0.6020**, MAE **11.4576**, `mae_gap` **0.476** | **0.6020 / 11.4576 / 0.4762**, n=1321 (incl. play-in + playoffs) | Yes | `p3_wf.py`: my own monthly walk-forward loop and metrics (A's `fit`/`predict_rows` only). Train = eligible final games dated before the 1st of the test month. |
| Early segment 2024-25 (either team <15 gp): window+Elo prior **0.6067** vs no prior **0.6554** | **0.6067 / 0.6554** (n=235). Margin MAE 9.57 vs 10.62. Rest of season 0.6056 vs 0.6070 | Yes | same harness, with `WindowParams(elo_prior_games=0, fit_elo_prior=False)` for the no-prior run |
| Flagged-side 1-unit ROI on 2025-26 negative for every model, −3.8% to −9.3% | ml −3.81%, home_always −6.27%, elo −7.46%, window −8.72%, window_noprior −9.30% | Yes (from the backtest rows) | Elo recomputed by hand below |
| — Elo by hand (edge ≥ 0.03, closing prices) | **2060 flags, hit rate 0.4820, ROI −0.0746**: identical to `backtest_results` | Yes | `p3_flags_elo.py`: my devig, Φ spread model, σ = RMS(margin − elo_margin) on eligible games before each month, settlement, unit return |

Elo flags by market (from the hand recompute): ML 1034 flags, hit 47.0%, ROI −9.3% (home side 864 at −7.5%, away side 170 at −18.6%). Spread 1026 flags, hit 49.4%, ROI −5.6%. **2060 flags on 1306 games**: at edge ≥ 0.03, Elo disagrees with the market by 3 or more points on about 80% of ML markets. A threshold that fires that often is measuring the model's error, not real edges (see Next).

**Is the Elo prior a leak? No.**
- `game_features.elo_margin` equals my independent pre-game Elo margin `(R_h − R_a + 100)/28` on all 4156 games (max |diff| = 0.0). Script: `p3_elo_leak.py`.
- Perturbation test: I added 40 points to the home score of `0022500500` and re-ran Elo. That game's own `elo_margin` is unchanged (0.830694 before and after). The home team's next game moves (1.021 → 2.181). So a game's result only reaches later games.
- **Does the prior actually fix early season?** Yes. Early-segment log loss and MAE:

| season, early segment | n | window + prior | window, no prior | Elo alone | market |
|---|---|---|---|---|---|
| 2024-25 | 235 | **0.6067** / 9.57 | 0.6554 / 10.62 | 0.6127 / 9.71 | (no lines) |
| 2025-26 | 232 | **0.5861** / 10.59 | 0.5998 / 11.04 | 0.5988 / 10.81 | **0.5625** / 10.27 |

  The blend beats both of its parts early in the season. The market is still better. The fitted prior strength k is about 17–18 games, so a prior weight of 0.5 holds at about 17 games played.
- **One caveat (new bug below):** Elo's home advantage of 100 points is too large. `elo_margin` averages **+3.66** against an actual home margin of +1.88 (2024-25) and +3.68 vs +1.77 (2025-26). So the prior pushes an extra ~1.8 points toward the home team. The window fit compensates: its raw hca drops to 0.19 with the prior, vs 0.91 without.

## 2. Line movement (250 open snapshots)

**Which games have opens? A biased sample.**

```sql
select g.season, g.season_type, o.book, min(g.game_date), max(g.game_date), count(*)
from odds_snapshots o join games g using(game_id) where o.is_closing=0 group by 1,2,3;
-- 2025-26 | regular | ESPN BET | 2025-10-21 | 2025-11-23 | 250
```

- All 250 are the core-API fallback games, i.e. exactly the games where ESPN's pickcenter was empty. They are 2025-26 regular season only, opening night through 2025-11-23 (ISO weeks 42–46, 42–53 games per week).
- The book is ESPN BET for both open and close. 231 of the 250 are in the "early" segment (max 17 games played).
- `espn_core_checked=1` holds for exactly these 250 games, and 250 of 250 have an open. No pickcenter (Draft Kings) game has an open, because `--with-open` was never run.
- The selection bias: early season, one book, and smaller spreads than the rest of the season (mean |close spread| 6.54 vs 7.38). Nothing here generalises to DK or to mid-season.

```sql
select case when exists(select 1 from odds_snapshots x where x.game_id=g.game_id and x.is_closing=0) then 'has open' else 'no open' end,
  count(*), round(avg(abs(c.home_spread)),2), round(avg(abs(g.home_pts-g.away_pts+c.home_spread)),3), round(100.0*avg(g.home_pts>g.away_pts),1)
from games g join odds_snapshots c on c.game_id=g.game_id and c.is_closing=1
where g.season='2025-26' and g.season_type='regular' group by 1;
-- has open 250 | 6.54 | 10.192 | 56.8      no open 979 | 7.38 | 11.063 | 55.2
```

**Size and direction** (`p3_move.py`; move = close − open on the home spread, so negative means it moved toward home):

```sql
with m as (select g.home_pts-g.away_pts margin, g.home_pts+g.away_pts tot, o.home_spread os, c.home_spread cs, o.total ot, c.total ct
   from odds_snapshots o join odds_snapshots c on c.game_id=o.game_id and c.is_closing=1 and c.source='espn'
   join games g on g.game_id=o.game_id where o.is_closing=0)
select count(*), sum(cs<os), sum(cs>os), sum(cs=os), round(avg(abs(cs-os)),2), max(abs(cs-os)), sum(abs(cs-os)>=2),
  sum(ct>ot), sum(ct<ot), sum(ct=ot), round(avg(abs(ct-ot)),2), max(abs(ct-ot)),
  round(avg(abs(margin+os)),3), round(avg(abs(margin+cs)),3), round(avg(abs(tot-ot)),3), round(avg(abs(tot-ct)),3) from m;
```

| | n | toward home | toward away | no move | mean \|move\| | max | moves ≥ 2 |
|---|---|---|---|---|---|---|---|
| spread | 250 | 72 | 115 | 63 | 1.44 | 9.0 | 84 |
| total | 250 | up 122 | down 92 | flat 36 | 2.30 | 11.0 | — |

- The mean spread move is +0.40 (sd 2.05), i.e. away teams were bet up on average. The mean ML no-vig move is |0.036|.
- The large moves are consistent across markets: the ML moves with the spread. For example DEN @ MIN 2025-10-27 went −1.5 → +6.5 with ML −125 → +195, and CHI @ DET 2025-11-12 went −5.5 → +3.5. That pattern looks like injury or news moves, not parse errors. Only 2 open rows have the ML and spread favourites disagreeing, and 0 are pick'em.

**Did the side the line moved toward cover?** (187 games where the spread moved)

| graded against | record | % | binomial p |
|---|---|---|---|
| open line | 100-87-0 | 53.5 | 0.380 |
| close line | 90-97-0 | 48.1 | 0.661 |
| open line, \|move\| ≥ 2 (n=84) | 48-36-0 | 57.1 | — |
| close line, \|move\| ≥ 2 | 43-41-0 | 51.2 | — |
| total moved toward (over if up), vs open | 112-102-0 | 52.3 | 0.539 |
| total moved toward, vs close | 99-115-0 | 46.3 | 0.305 |

This is the textbook pattern: the move side beats the stale open a little and is a coin flip against the close. None of it is significant at this sample size.

**Does the model agree with the move direction?** "Model prefers home vs the open" means `pred_margin > −open_spread`. Window predictions are my walk-forward ones; Elo is `elo_margin`.

| model | agrees with spread-move direction | binomial p | corr(model − open, move toward home) | p |
|---|---|---|---|---|
| window | 101/187 = **54.0%** | 0.31 | **0.285** | **0.0001** |
| elo | 91/187 = 48.7% | 0.77 | 0.162 | 0.027 |
| window total | 110/214 = 51.4% | 0.73 | — | — |

The direction match is not better than chance. But the *size* of the window model's disagreement with the open correlates with how far the line then moved (r = 0.285). When the model is far from the open, the market tends to move toward the model. That's a weak but real signal that the model sees some of what the market later prices in. It would be worth re-testing on DK opens (see Next).

**Is the open or the close more accurate?** (n=250)

| | open | close | window model | Elo |
|---|---|---|---|---|
| margin MAE | 10.196 | 10.192 | 10.439 | 10.668 |
| total MAE | 14.248 | 14.248 | 14.812 | — |
| ML log loss | 0.5674 | **0.5631** | — | — |

MAE(open) − MAE(close) = 0.004, 95% bootstrap CI [−0.252, 0.260]. On spreads they are indistinguishable even on the moved games (10.340 vs 10.334) and the |move| ≥ 2 games (11.083 vs 11.024). The close is slightly sharper on the moneyline. Both beat the window model.

## 3. Calibration on 2025-26

The common sample is 1306 final games with a closing ML. Window predictions are my walk-forward ones. Script: `p3_calib.py`.

| model | log loss | Brier | ECE (10 bins) | sd(p) | recalibration intercept / slope |
|---|---|---|---|---|---|
| market (no-vig) | **0.5746** | **0.1965** | **0.0290** | 0.215 | −0.002 / 1.076 |
| window | 0.6012 | 0.2070 | 0.0303 | 0.184 | −0.005 / 1.114 |
| elo | 0.6109 | 0.2103 | 0.0668 | 0.194 | **−0.326** / 0.991 |

Reliability (mean predicted p → actual home-win rate, n):

| bin | market | window | elo |
|---|---|---|---|
| 0.1–0.2 | 0.156 → 0.130 (69) | 0.168 → 0.053 (38) | 0.166 → 0.000 (21) |
| 0.2–0.3 | 0.253 → 0.204 (103) | 0.256 → 0.231 (91) | 0.257 → 0.171 (70) |
| 0.3–0.4 | 0.351 → 0.371 (151) | 0.352 → 0.312 (157) | 0.356 → 0.324 (102) |
| 0.4–0.5 | 0.449 → 0.490 (194) | 0.453 → 0.448 (210) | 0.452 → 0.349 (172) |
| 0.5–0.6 | 0.556 → 0.532 (188) | 0.549 → 0.620 (263) | 0.554 → 0.500 (190) |
| 0.6–0.7 | 0.651 → 0.672 (195) | 0.651 → 0.647 (241) | 0.653 → 0.602 (231) |
| 0.7–0.8 | 0.749 → 0.765 (200) | 0.747 → 0.766 (175) | 0.753 → 0.684 (247) |
| 0.8–0.9 | 0.848 → 0.819 (160) | 0.842 → 0.843 (108) | 0.849 → 0.781 (210) |
| 0.9–1.0 | 0.921 → 1.000 (30) | 0.915 → 0.800 (20) | 0.925 → 0.871 (62) |
| (0–0.1: 16 / 3 / 1 games) | | | |

- The **market** and **window** models are well calibrated. Their slopes are slightly above 1, so both are a touch *under*-confident. The window model is also less sharp (sd 0.184 vs 0.215).
- **Elo's predictions are too high in every bin** (intercept −0.33). The mean `elo_prob` is 0.621 against a home-win rate of 0.555. That is the home-advantage bias from §1, not a sharpness problem (slope 0.99).
- Sweeping Elo `home_adv` (`p3_elo_hca.py`, independent Elo, 2024-25 + 2025-26):

| home_adv | 40 | 50 | 60 | 70 | 80 | 100 (current) |
|---|---|---|---|---|---|---|
| log loss | 0.6060 | 0.6063 | 0.6075 | 0.6093 | 0.6118 | 0.6191 |
| mean margin bias | −0.35 | +0.02 | +0.38 | +0.75 | +1.11 | +1.85 |

  This sweep is in-sample on the test seasons, so read it as "100 is clearly too high". It doesn't choose a value; tune on 2023-24.

## 4. Where the window model is closest to the market

This uses my 2025-26 walk-forward window predictions on 1311 games with a spread. gap = window MAE − market MAE, with a 95% game-bootstrap CI. Script: `p3_close.py`.

| subset | n | market MAE | window MAE | Elo MAE | gap [95% CI] |
|---|---|---|---|---|---|
| regular | 1228 | 10.886 | 11.393 | 11.520 | 0.507 [0.314, 0.705] |
| playoffs | 77 | 13.097 | 13.098 | 13.294 | **0.000** [−0.818, 0.815] |
| early (< 15 gp) | 232 | 10.272 | 10.586 | 10.813 | **0.315** [−0.140, 0.761] |
| rest | 1079 | 11.188 | 11.699 | 11.806 | 0.511 [0.319, 0.699] |
| 2025-10 | 80 | 10.812 | **10.710** | 10.808 | **−0.103** [−0.994, 0.789] |
| 2025-11 | 217 | 9.873 | 10.291 | 10.483 | 0.418 [0.021, 0.826] |
| 2025-12 | 197 | 11.612 | 11.832 | 11.673 | 0.220 [−0.199, 0.631] |
| 2026-01 | 233 | 11.036 | 11.668 | 11.558 | 0.632 [0.157, 1.055] |
| 2026-02 | 166 | 11.416 | 12.181 | 12.745 | 0.766 [0.241, 1.293] |
| 2026-03 | 239 | 10.688 | 11.157 | 11.191 | 0.469 [0.048, 0.894] |
| 2026-04 | 140 | 11.464 | 12.361 | 12.654 | 0.896 [0.158, 1.621] |
| \|spread\| 0.5–3 | 271 | 12.574 | 12.607 | 12.597 | **0.033** [−0.340, 0.435] |
| \|spread\| 3.5–6.5 | 457 | 11.321 | 11.967 | 11.928 | 0.646 [0.336, 0.935] |
| \|spread\| 7–9.5 | 249 | 9.187 | 9.719 | 9.850 | 0.533 [0.114, 0.968] |
| \|spread\| 10+ | 334 | 10.737 | 11.298 | 11.764 | 0.562 [0.130, 0.978] |
| rest adv ≤ −2 / −1 / 0 / +1 / ≥ +2 | 43 / 223 / 742 / 241 / 62 | | | | 0.96 / 0.26 / 0.47 / 0.60 / 0.58 |
| neither b2b | 941 | 10.779 | 11.125 | 11.264 | 0.346 [0.135, 0.548] |
| home b2b only | 145 | 12.162 | 12.839 | 12.967 | 0.677 [0.048, 1.314] |
| away b2b only | 155 | 11.655 | 12.388 | 12.452 | 0.733 [0.115, 1.305] |
| both b2b | 70 | 10.586 | 11.832 | 11.965 | 1.246 [0.540, 1.897] |

- The model is closest to the market early in the season and in October 2025. Those are the core-API ESPN BET games: the prior works, and that book's close is a little softer.
- It is also close in the playoffs and in near-pick'em games (|spread| ≤ 3, where nobody predicts well).
- It is furthest away late in the regular season (April: +0.90, when rotations and tanking change) and when both teams are on a back-to-back (+1.25). Rest and travel news that the market prices and the model's fixed `b2b_pts` doesn't capture would explain that.
- These CIs exclude 0 in favour of the model nowhere.

## 5. Today's slate (2026-11-15) vs ESPN lines

`nba daily` wrote 27 predictions (9 games × elo/window/ml) at `2026-11-15T23:59:23Z`. By then IND @ PHI was already final in the DB and several games had tipped. The predictions only use games before the date, so that isn't a leak. But predictions made after tip-off should be marked as such. The scoreboard `odds` field (`curl …/scoreboard?dates=20261115`) only carried lines for the 3 games not yet started (HOU @ WSH, ATL @ MIL, OKC @ SAC). Its moneyline sits under `odds[0].moneyline.home.close.odds`, not `homeTeamOdds.moneyLine`. For all 9 games I used the pregame DK line from each `summary?event=` pickcenter (read-only, 9 calls). Script: `p3_slate.py`.

| game | DK home margin (−spread) | window | elo | ml | DK no-vig p(home) | window p | DK total | window total | ml total | final margin |
|---|---|---|---|---|---|---|---|---|---|---|
| IND @ PHI | +7.5 | +7.40 | +10.66 | +15.84 | 0.713 | 0.697 | 226.5 | 223.8 | 239.3 | +7 |
| LAL @ PHX | −3.5 | −3.53 | −0.89 | −4.42 | 0.407 | 0.403 | 226.5 | 229.5 | 229.8 | −8 |
| CHA @ LAC | +1.5 | +4.77 | +3.65 | +2.05 | 0.520 | 0.630 | 214.5 | 218.1 | 230.0 | (+8, ESPN) |
| SAS @ UTA | −14.5 | −16.12 | −12.83 | −15.35 | 0.121 | 0.131 | 222.5 | 224.1 | 227.2 | live |
| DAL @ BOS | +14.5 | +12.41 | +12.32 | +13.34 | 0.870 | 0.806 | 217.5 | 222.2 | 222.7 | live |
| NYK @ TOR | −4.5 | −1.32 | −2.43 | +0.23 | 0.391 | 0.463 | 218.5 | 219.5 | 225.4 | live |
| HOU @ WAS | −8.5 | −9.91 | −7.42 | −10.36 | 0.259 | 0.245 | 224.5 | 223.1 | 238.5 | — |
| ATL @ MIL | −8.5 | −5.42 | −4.12 | −8.51 | 0.260 | 0.353 | 233.5 | 229.0 | 216.2 | — |
| OKC @ SAC | −11.5 | −11.46 | −8.49 | −15.52 | 0.208 | 0.212 | 223.5 | 224.1 | 226.6 | — |

Summary versus DK: window mean |margin gap| **1.65** (max 3.27, corr 0.974), mean |p gap| 0.043, mean |total gap| 2.58. Elo 2.48 (max 4.38). ML 2.49 (**max 8.34**, IND @ PHI +15.8), with ML totals 9.19 off on average (216 to 239). The window and Elo predictions are sane. The ML model's totals and extremes are not, which matches its backtest totals (MAE 18.48, worse than predicting the season mean at 16.06) and its early-season log loss of 0.818. C is already working on the ML model.

## Next

- **Bug (A/C): Elo `home_adv = 100` is too large.** It adds +1.85 points of home bias to `elo_margin`, so `elo_prob` is too high in every calibration bin (intercept −0.33). The early-season prior inherits the bias. Tune `EloParams.home_adv` on 2023-24 only; 50–60 looks right (log loss 0.619 → ~0.606 on 2024-26 in-sample). Then re-fit the window prior. Test: the mean `elo_margin` over a season should be within ±0.5 of the actual mean home margin.
- **Feature (dashboard): a "Line movement" view.** Per game, show open → close spread and total, size and direction of the move, whether the move side covered against open and close, and the model's gap to the open. Include the selection caveat: the 250 games are ESPN BET only, 2025-10-21..11-23. Then run `nba espn --with-open` over 2025-11-24..2026-04-12 to get DK-era opens. The r = 0.285 between model − open and the move is the result worth re-testing on a non-early, single-book sample.
- **Feature (flags): calibrate the flag threshold to the model's error, not a fixed 0.03.** At 0.03, Elo flags 2060 of about 2600 possible ML/spread markets on 1306 games, and every model's flagged sides lose 4–9% at closing prices. A dashboard "edge vs result" table (hit rate by edge bucket, per model), next to a threshold slider, would show where, if anywhere, the disagreement starts to mean anything.
- **Small data gap:** `0022500283` (DET @ IND, 2025-11-24, event 401810138) still has no close and `espn_core_checked = 0`. One `nba espn --from 2025-11-24 --to 2025-11-24` run should fill it via the core API, making the regular season 1230/1230.
- **Small (A): stamp `predictions` with "made after tip-off"** when `created_at` is later than the game's start, e.g. the 2026-11-15 run after IND @ PHI was final. Then dashboard grading can exclude these games from any live-performance view.

## Appendix: scripts (verbatim)

<details><summary><code>p3_verify_market.py</code></summary>

```python
"""Market log loss / margin MAE on 2025-26 from raw closing snapshots (no nba_predict code)."""
import sqlite3, sys, numpy as np, pandas as pd
db = sqlite3.connect(sys.argv[1])
d = pd.read_sql("""select g.game_id, g.season_type, g.home_pts-g.away_pts margin, (g.home_pts>g.away_pts)*1.0 y,
   o.home_ml, o.away_ml, o.home_spread from games g join odds_snapshots o on o.game_id=g.game_id and o.source='espn' and o.is_closing=1
   where g.season='2025-26' and g.status='final'""", db)
imp = lambda a: np.where(a < 0, -a/(-a+100), 100/(a+100))
m = d.dropna(subset=["home_ml","away_ml"]).copy()
ph, pa = imp(m.home_ml.astype(float)), imp(m.away_ml.astype(float)); m["p"] = ph/(ph+pa)
ll = lambda p, y: float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
s = d.dropna(subset=["home_spread"])
print("ML games", len(m), "market ll %.4f" % ll(m.p, m.y), "| spread games", len(s), "MAE %.4f" % (s.margin + s.home_spread).abs().mean())
for t, x in m.groupby("season_type"): print("  ", t, len(x), round(ll(x.p, x.y), 4))
```
</details>

<details><summary><code>p3_wf.py</code></summary>

```python
"""Independent monthly walk-forward harness (my loop + my metrics; A's window fit/predict only).
Train = final games dated before the 1st of the test month, excluding 'ineligible' rows
(a side with <3 season games and no prior-season data). Test = every final game of the month."""
import sqlite3, sys, numpy as np, pandas as pd
from nba_predict.models.window_rating import fit, predict_rows, WindowParams
db = sqlite3.connect(sys.argv[1])
f = pd.read_sql("select * from game_features", db)
fin = f[(f.status == "final") & f.home_win.notna()].copy()
elig = ~(((fin.h_gp_szn < 3) & (fin.h_all_l30_gp <= fin.h_gp_szn)) | ((fin.a_gp_szn < 3) & (fin.a_all_l30_gp <= fin.a_gp_szn)))
train_pool = fin[elig]
ll = lambda p, y: -(y*np.log(np.clip(p,1e-15,1-1e-15)) + (1-y)*np.log(np.clip(1-p,1e-15,1)))
out = []
for season in ["2024-25", "2025-26"]:
    sf = fin[fin.season == season]
    for name, base in [("window", WindowParams()), ("window_noprior", WindowParams(elo_prior_games=0.0, fit_elo_prior=False))]:
        preds = []
        for mo in sorted(sf.game_date.str[:7].unique()):
            tr = train_pool[train_pool.game_date < f"{mo}-01"]; te = sf[sf.game_date.str[:7] == mo]
            p = fit(tr, base); preds.append(predict_rows(te, p).assign(k=p.elo_prior_games))
        d = sf.merge(pd.concat(preds), on="game_id")
        d["early"] = (d.h_gp_szn < 15) | (d.a_gp_szn < 15)
        for seg, x in [("all", d), ("early", d[d.early]), ("rest", d[~d.early])]:
            sp = x[x.m_home_spread.notna()]
            out.append(dict(season=season, model=name, seg=seg, n=len(x), log_loss=ll(x.home_win_prob, x.home_win).mean(),
                            margin_mae=(x.pred_margin - x.margin).abs().mean(),
                            mae_gap=((sp.pred_margin - sp.margin).abs() - (sp.margin + sp.m_home_spread).abs()).mean(),
                            mean_k=x.k.mean()))
        d.to_pickle(f"{sys.argv[2]}/wf_{name}_{season}.pkl")
print(pd.DataFrame(out).round(4).to_string(index=False))
```
</details>

<details><summary><code>p3_elo_leak.py</code></summary>

```python
"""Leak checks for the Elo prior. (1) independent pre-game Elo margin == game_features.elo_margin.
(2) perturbation: change one final game's score, recompute: that game's own elo_margin must not move."""
import sqlite3, sys, numpy as np, pandas as pd
db = sqlite3.connect(sys.argv[1])
g0 = pd.read_sql("select game_id, season, game_date, home_team_id h, away_team_id a, home_pts hp, away_pts ap, status from games order by game_date, game_id", db)
def elo(g):
    R, last, out = {}, {}, {}
    for r in g.itertuples():
        for t in (r.h, r.a):
            if t not in R: R[t] = 1505.0
            elif last[t] != r.season: R[t] = 1505 + 0.75*(R[t]-1505)
            last[t] = r.season
        d = R[r.h] - R[r.a] + 100; p = 1/(1+10**(-d/400)); out[r.game_id] = (d/28.0, p)
        if r.status != "final": continue
        mov = r.hp - r.ap; hw = mov > 0
        delta = 20*((abs(mov)+3)**0.8/(7.5+0.006*(d if hw else -d)))*((1.0 if hw else 0.0)-p)
        R[r.h] += delta; R[r.a] -= delta
    return out
base = elo(g0)
gf = pd.read_sql("select game_id, elo_margin, elo_prob from game_features", db)
gf["mine"] = gf.game_id.map(lambda x: base[x][0])
print("max |elo_margin - independent| =", float((gf.elo_margin - gf.mine).abs().max()), "n", len(gf))
gid = "0022500500"
g1 = g0.copy(); i = g1.index[g1.game_id == gid][0]; g1.loc[i, "hp"] += 40
pert = elo(g1)
print("perturbed game", gid, "own elo_margin before/after:", round(base[gid][0], 6), round(pert[gid][0], 6))
nxt = g0[(g0.game_date > g0.loc[i, "game_date"]) & ((g0.h == g0.loc[i, "h"]) | (g0.a == g0.loc[i, "h"]))].iloc[0].game_id
print("home team's next game", nxt, "moves:", round(base[nxt][0], 3), "->", round(pert[nxt][0], 3))
```
</details>

<details><summary><code>p3_flags_elo.py</code></summary>

```python
"""Hand recompute of Elo flagged sides, 2025-26 (no nba_predict code). Rule: per market, side with the
higher edge = model_prob - closing no-vig prob, kept if edge >= 0.03; 1 unit at the closing price.
Spread model prob = Phi((elo_margin + home_spread)/sigma), sigma = RMS(margin - elo_margin) over
eligible final games before the test month. Totals: Elo has no total -> none."""
import sqlite3, sys, numpy as np, pandas as pd
from scipy.stats import norm
db = sqlite3.connect(sys.argv[1])
f = pd.read_sql("""select game_id, season, game_date, status, home_win, margin, elo_prob, elo_margin,
  h_gp_szn, a_gp_szn, h_all_l30_gp, a_all_l30_gp from game_features""", db)
s = pd.read_sql("""select game_id, home_ml, away_ml, home_spread, home_spread_price hsp, away_spread_price asp
  from odds_snapshots where source='espn' and is_closing=1""", db)
g = pd.read_sql("select game_id, home_pts, away_pts from games", db)
fin = f[(f.status == "final") & f.home_win.notna()]
elig = ~(((fin.h_gp_szn < 3) & (fin.h_all_l30_gp <= fin.h_gp_szn)) | ((fin.a_gp_szn < 3) & (fin.a_all_l30_gp <= fin.a_gp_szn)))
pool = fin[elig]
imp = lambda a: -a/(-a+100) if a < 0 else 100/(a+100)
ret = lambda price, res: 0.0 if res == "push" else (-1.0 if res == "loss" else (100/-price if price < 0 else price/100))
d = fin[fin.season == "2025-26"].merge(s, on="game_id").merge(g, on="game_id")
d = d[d.home_ml.notna() & d.away_ml.notna()]            # backtest evaluates flags on games with an ML no-vig
sig = {mo: float(np.sqrt(np.mean((pool[pool.game_date < f"{mo}-01"].margin - pool[pool.game_date < f"{mo}-01"].elo_margin) ** 2)))
       for mo in d.game_date.str[:7].unique()}
bets = []
for r in d.itertuples():
    ph, pa = imp(r.home_ml), imp(r.away_ml); nh = ph/(ph+pa)
    cands = [("ml", "home", r.elo_prob - nh, r.home_ml, r.margin), ("ml", "away", (1-r.elo_prob) - (1-nh), r.away_ml, -r.margin)]
    best = max(cands, key=lambda c: c[2]); picks = [best] if best[2] >= 0.03 else []
    if pd.notna(r.home_spread) and pd.notna(r.hsp) and pd.notna(r.asp):
        sh, sa = imp(r.hsp), imp(r.asp); nsh = sh/(sh+sa)
        p = norm.cdf((r.elo_margin + r.home_spread) / sig[r.game_date[:7]])
        c = [("spread", "home", p - nsh, r.hsp, r.margin + r.home_spread), ("spread", "away", (1-p) - (1-nsh), r.asp, -r.margin - r.home_spread)]
        b = max(c, key=lambda x: x[2])
        if b[2] >= 0.03: picks.append(b)
    for mkt, side, edge, price, diff in picks:
        res = "win" if diff > 0 else "loss" if diff < 0 else "push"
        bets.append(dict(game_id=r.game_id, market=mkt, side=side, edge=edge, price=price, res=res, units=ret(price, res)))
b = pd.DataFrame(bets)
wl = b[b.res != "push"]
print("bets", len(b), "hit_rate %.4f" % (wl.res == "win").mean(), "roi %.4f" % b.units.mean())
print(b.groupby("market").agg(n=("units","size"), hit=("res", lambda x: (x=="win").mean()), roi=("units","mean")).round(4).to_string())
print(b.groupby(["market","side"]).agg(n=("units","size"), roi=("units","mean")).round(4).to_string())
```
</details>

<details><summary><code>p3_move.py</code></summary>

```python
"""Line movement on the 250 open/close pairs (ESPN BET, 2025-10-21..11-23). Window model = my walk-forward
predictions (A's fit, monthly refit); Elo = elo_margin. Move = close - open (home spread; <0 = toward home)."""
import sqlite3, sys, numpy as np, pandas as pd
from scipy.stats import binomtest
S = sys.argv[2]; db = sqlite3.connect(sys.argv[1])
d = pd.read_sql("""select g.game_id, g.game_date, g.home_pts-g.away_pts margin, g.home_pts+g.away_pts tot,
  o.home_spread os, c.home_spread cs, o.total ot, c.total ct, o.home_ml ohm, o.away_ml oam, c.home_ml chm, c.away_ml cam
  from odds_snapshots o join odds_snapshots c on c.game_id=o.game_id and c.is_closing=1 and c.source='espn'
  join games g on g.game_id=o.game_id where o.is_closing=0""", db)
w = pd.read_pickle(f"{S}/wf_window_2025-26.pkl")[["game_id", "pred_margin", "pred_total", "elo_margin", "h_gp_szn", "a_gp_szn"]]
d = d.merge(w, on="game_id")
imp = lambda a: np.where(a < 0, -a/(-a+100), 100/(a+100))
d["p_open"] = imp(d.ohm)/(imp(d.ohm)+imp(d.oam)); d["p_close"] = imp(d.chm)/(imp(d.chm)+imp(d.cam))
d["mv"] = d.cs - d.os; d["tmv"] = d.ct - d.ot
print("n", len(d), "| mean move %.2f (sd %.2f) | mean |move| %.2f | ML no-vig home prob move mean |%.3f|" % (d.mv.mean(), d.mv.std(), d.mv.abs().mean(), (d.p_close-d.p_open).abs().mean()))
print("spread move distribution (close-open):"); print(d.mv.value_counts().sort_index().to_string())
print("total move distribution:"); print(pd.cut(d.tmv, [-20,-4.5,-2.5,-0.5,0.5,2.5,4.5,20]).value_counts().sort_index().to_string())
def cover_toward(x, line):
    side = np.where(x.mv < 0, 1, -1)  # +1 = home side was moved toward
    diff = side * (x.margin + x[line]); return (diff > 0).sum(), (diff < 0).sum(), (diff == 0).sum()
mvd = d[d.mv != 0]
for line in ["os", "cs"]:
    w_, l_, p_ = cover_toward(mvd, line)
    print(f"side moved toward, graded vs {'open' if line=='os' else 'close'} line: {w_}-{l_}-{p_}  ({100*w_/(w_+l_):.1f}%, binom p={binomtest(w_, w_+l_).pvalue:.3f})")
big = mvd[mvd.mv.abs() >= 2]
for line in ["os", "cs"]:
    w_, l_, p_ = cover_toward(big, line); print(f"  |move|>=2 vs {line}: {w_}-{l_}-{p_} ({100*w_/(w_+l_):.1f}%)")
tm = d[d.tmv != 0]
for line in ["ot", "ct"]:
    diff = np.sign(tm.tmv) * (tm.tot - tm[line]); w_, l_ = (diff > 0).sum(), (diff < 0).sum()
    print(f"total moved toward (over if up), graded vs {line}: {w_}-{l_}-{(diff==0).sum()} ({100*w_/(w_+l_):.1f}%, p={binomtest(w_, w_+l_).pvalue:.3f})")
# model agreement with move direction: model prefers home vs OPEN line  <=> pred_margin > -os ; move toward home <=> mv < 0
for name, col in [("window", "pred_margin"), ("elo", "elo_margin")]:
    x = mvd[mvd[col] != -mvd.os]
    agree = ((x[col] > -x.os) == (x.mv < 0)); k = int(agree.sum())
    print(f"{name}: agrees with spread move direction {k}/{len(x)} = {100*k/len(x):.1f}% (binom p={binomtest(k, len(x)).pvalue:.4f})")
    print(f"   corr(model - open implied, move toward home = -(close-open)) = {np.corrcoef(x[col] + x.os, -x.mv)[0,1]:.3f}")
x = tm[tm.pred_total.notna() & (tm.pred_total != tm.ot)]
agree = ((x.pred_total > x.ot) == (x.tmv > 0)); k = int(agree.sum())
print(f"window total: agrees with total move {k}/{len(x)} = {100*k/len(x):.1f}% (p={binomtest(k, len(x)).pvalue:.4f})")
print("MAE vs margin: open %.3f close %.3f | window %.3f elo %.3f   (n=%d)" % ((d.margin + d.os).abs().mean(), (d.margin + d.cs).abs().mean(), (d.pred_margin - d.margin).abs().mean(), (d.elo_margin - d.margin).abs().mean(), len(d)))
print("moved games only (n=%d): open %.3f close %.3f" % (len(mvd), (mvd.margin + mvd.os).abs().mean(), (mvd.margin + mvd.cs).abs().mean()))
print("|move|>=2 (n=%d): open %.3f close %.3f" % (len(big), (big.margin + big.os).abs().mean(), (big.margin + big.cs).abs().mean()))
print("total MAE: open %.3f close %.3f window %.3f" % ((d.tot - d.ot).abs().mean(), (d.tot - d.ct).abs().mean(), (d.pred_total - d.tot).abs().mean()))
ll = lambda p, y: float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))); y = (d.margin > 0).astype(float)
print("ML log loss: open %.4f close %.4f" % (ll(d.p_open, y), ll(d.p_close, y)))
# paired bootstrap of MAE diff open-close
rng = np.random.default_rng(0); a = ((d.margin + d.os).abs() - (d.margin + d.cs).abs()).to_numpy()
bs = a[rng.integers(0, len(a), (5000, len(a)))].mean(1); print("MAE(open)-MAE(close) = %.3f, 95%% CI [%.3f, %.3f]" % (a.mean(), *np.percentile(bs, [2.5, 97.5])))
```
</details>

<details><summary><code>p3_calib.py</code></summary>

```python
"""Calibration (2025-26, all final games with a closing ML; walk-forward window, elo_prob, market no-vig)
and where the window model is closest to the market (mae_gap by month / |spread| / rest / early).
Also extends the walk-forward to 2026-27 (Oct, Nov) for the live season."""
import sqlite3, sys, numpy as np, pandas as pd
from scipy.stats import pearsonr
from nba_predict.models.window_rating import fit, predict_rows, WindowParams
S = sys.argv[2]; db = sqlite3.connect(sys.argv[1])
# from move analysis is recomputed in p3_move.py; here: n=187 -> p for r
for r in (0.285, 0.162):
    n = 187; t = r*np.sqrt((n-2)/(1-r*r)); from scipy.stats import t as T; print("r=%.3f n=%d p=%.5f" % (r, n, 2*T.sf(t, n-2)))
w = pd.read_pickle(f"{S}/wf_window_2025-26.pkl")
imp = lambda a: np.where(a < 0, -a/(-a+100), 100/(a+100))
s = pd.read_sql("select game_id, home_ml, away_ml from odds_snapshots where source='espn' and is_closing=1", db)
d = w.drop(columns=["m_home_ml","m_away_ml"], errors="ignore").merge(s, on="game_id")
d = d[d.home_ml.notna() & d.away_ml.notna()].copy()
d["p_mkt"] = imp(d.home_ml)/(imp(d.home_ml)+imp(d.away_ml))
bins = np.linspace(0, 1, 11)
rows = []
for name, col in [("market", "p_mkt"), ("window", "home_win_prob"), ("elo", "elo_prob")]:
    b = pd.cut(d[col], bins, include_lowest=True).rename("bin")
    t = d.groupby(b, observed=True).agg(n=("home_win", "size"), mean_p=(col, "mean"), actual=("home_win", "mean"))
    t["model"] = name; rows.append(t.reset_index()[["bin","n","mean_p","actual","model"]])
    e = (t.n/t.n.sum()*(t.mean_p-t.actual).abs()).sum()
    ll = float(-np.mean(d.home_win*np.log(d[col])+(1-d.home_win)*np.log(1-d[col])))
    print(f"{name}: n={len(d)} ECE={e:.4f} logloss={ll:.4f} brier={np.mean((d[col]-d.home_win)**2):.4f} sd(p)={d[col].std():.3f}")
cal = pd.concat(rows)
print(cal.pivot_table(index="bin", columns="model", values=["n", "mean_p", "actual"], observed=True).round(3).to_string())
# slope: logit(actual) ~ logit(p)
from numpy.linalg import lstsq
for name, col in [("market","p_mkt"),("window","home_win_prob"),("elo","elo_prob")]:
    x = np.log(d[col]/(1-d[col]))
    # logistic recalibration slope via simple IRLS
    X = np.column_stack([np.ones(len(x)), x]); bta = np.zeros(2)
    for _ in range(25):
        p = 1/(1+np.exp(-X@bta)); W = p*(1-p); bta += np.linalg.solve(X.T@(X*W[:,None]), X.T@(d.home_win-p))
    print(f"{name}: recalibration intercept {bta[0]:+.3f} slope {bta[1]:.3f}  (1.0 = calibrated; <1 = overconfident)")
```
</details>

<details><summary><code>p3_elo_hca.py</code></summary>

```python
"""Independent Elo with varying home_adv; log loss / margin bias on 2024-25+2025-26 final games (all types)."""
import sqlite3, sys, numpy as np, pandas as pd
g = pd.read_sql("select game_id, season, game_date, home_team_id h, away_team_id a, home_pts hp, away_pts ap, status from games order by game_date, game_id", sqlite3.connect(sys.argv[1]))
def run(hca):
    R, last, P, M = {}, {}, [], []
    for r in g.itertuples():
        for t in (r.h, r.a):
            if t not in R: R[t] = 1505.0
            elif last[t] != r.season: R[t] = 1505 + 0.75*(R[t]-1505)
            last[t] = r.season
        d = R[r.h]-R[r.a]+hca; p = 1/(1+10**(-d/400)); P.append(p); M.append(d/28)
        if r.status != "final": continue
        mov = r.hp-r.ap; hw = mov > 0
        delta = 20*((abs(mov)+3)**0.8/(7.5+0.006*(d if hw else -d)))*((1.0 if hw else 0.0)-p); R[r.h] += delta; R[r.a] -= delta
    x = g.assign(p=P, m=M); x = x[(x.status == "final") & x.season.isin(["2024-25","2025-26"])]
    y = (x.hp > x.ap).astype(float); mg = x.hp - x.ap
    return -np.mean(y*np.log(x.p)+(1-y)*np.log(1-x.p)), (x.m - mg).abs().mean(), x.m.mean() - mg.mean()
for h in (40, 50, 60, 70, 80, 100):
    ll, mae, bias = run(h); print(f"home_adv {h:3d}: logloss {ll:.4f}  margin MAE {mae:.3f}  mean bias {bias:+.2f}")
```
</details>

<details><summary><code>p3_close.py</code></summary>

```python
"""Elo home bias + where the window model is closest to the market (2025-26 walk-forward, games with a spread)."""
import sqlite3, sys, numpy as np, pandas as pd
S = sys.argv[2]; db = sqlite3.connect(sys.argv[1])
w = pd.read_pickle(f"{S}/wf_window_2025-26.pkl")
for seas in ["2024-25", "2025-26"]:
    x = pd.read_pickle(f"{S}/wf_window_{seas}.pkl")
    print(f"{seas}: mean elo_prob {x.elo_prob.mean():.3f} vs home win rate {x.home_win.mean():.3f}; mean elo_margin {x.elo_margin.mean():+.2f} vs actual {x.margin.mean():+.2f}; window pred {x.pred_margin.mean():+.2f}")
d = w[w.m_home_spread.notna()].copy()
d["mkt"] = -d.m_home_spread
d["ae_m"] = (d.pred_margin - d.margin).abs(); d["ae_k"] = (d.mkt - d.margin).abs(); d["ae_e"] = (d.elo_margin - d.margin).abs()
d["early"] = np.where((d.h_gp_szn < 15) | (d.a_gp_szn < 15), "early (<15 gp)", "rest")
d["month"] = d.game_date.str[:7]
d["sb"] = pd.cut(d.m_home_spread.abs(), [0, 3, 6.5, 9.5, 30], labels=["0.5-3", "3.5-6.5", "7-9.5", "10+"])
d["rest_adv"] = pd.cut((d.h_rest - d.a_rest).clip(-3, 3), [-4, -2, -1, 0, 1, 3], labels=["<=-2", "-1", "0", "+1", ">=+2"])
d["b2b"] = np.select([(d.h_b2b == 1) & (d.a_b2b == 0), (d.h_b2b == 0) & (d.a_b2b == 1), (d.h_b2b == 1) & (d.a_b2b == 1)], ["home b2b", "away b2b", "both"], "neither")
rng = np.random.default_rng(0)
def summ(x):
    g = (x.ae_m - x.ae_k).to_numpy(); bs = g[rng.integers(0, len(g), (2000, len(g)))].mean(1)
    return pd.Series(dict(n=len(x), mae_market=x.ae_k.mean(), mae_window=x.ae_m.mean(), mae_elo=x.ae_e.mean(), gap=g.mean(),
                          gap_lo=np.percentile(bs, 2.5), gap_hi=np.percentile(bs, 97.5), corr=x.pred_margin.corr(x.mkt),
                          mean_abs_disagree=(x.pred_margin - x.mkt).abs().mean()))
pd.set_option("display.width", 220)
for k in ["season_type", "early", "month", "sb", "rest_adv", "b2b"]:
    print(f"\n-- by {k}"); print(d.groupby(k, observed=True).apply(summ, include_groups=False).round(3).to_string())
```
</details>

<details><summary><code>p3_slate.py</code></summary>

```python
"""2026-11-15 predictions (nba daily) vs ESPN/DK pregame lines from summary pickcenter (read-only curl)."""
import json, sqlite3, sys, numpy as np, pandas as pd
S = sys.argv[2]; db = sqlite3.connect(sys.argv[1])
ev = {"0022600236": "401909985", "0022600237": "401909986", "0022600238": "401909987", "0022600239": "401909988", "0022600240": "401909989",
      "0022600241": "401909990", "0022600242": "401909991", "0022600243": "401909992", "0022600244": "401909993"}
imp = lambda a: -a/(-a+100) if a < 0 else 100/(a+100)
rows = []
for gid, e in ev.items():
    pc = json.load(open(f"{S}/raw3/sum{e}.json"))["pickcenter"][0]
    hm, am = pc["homeTeamOdds"]["moneyLine"], pc["awayTeamOdds"]["moneyLine"]
    rows.append(dict(game_id=gid, mkt_margin=-float(pc["spread"]), mkt_total=float(pc["overUnder"]), mkt_p=imp(hm)/(imp(hm)+imp(am))))
m = pd.DataFrame(rows)
p = pd.read_sql("select p.game_id, p.model, p.home_win_prob, p.pred_margin, p.pred_total from predictions p", db)
g = pd.read_sql("select game_id, status, home_pts-away_pts margin from games where game_date='2026-11-15'", db)
w = p.pivot(index="game_id", columns="model", values=["pred_margin", "home_win_prob", "pred_total"])
w.columns = [f"{a}_{b}" for a, b in w.columns]
d = m.merge(w.reset_index(), on="game_id").merge(g, on="game_id")
pd.set_option("display.width", 250)
cols = ["game_id", "status", "mkt_margin", "pred_margin_window", "pred_margin_elo", "pred_margin_ml", "mkt_p", "home_win_prob_window", "home_win_prob_elo", "home_win_prob_ml", "mkt_total", "pred_total_window", "pred_total_ml", "margin"]
print(d[cols].round(3).to_string(index=False))
for mdl in ["window", "elo", "ml"]:
    print(f"{mdl}: mean |margin - mkt| {np.abs(d[f'pred_margin_{mdl}'] - d.mkt_margin).mean():.2f}, max {np.abs(d[f'pred_margin_{mdl}'] - d.mkt_margin).max():.2f}, "
          f"mean |p - mkt_p| {np.abs(d[f'home_win_prob_{mdl}'] - d.mkt_p).mean():.3f}, corr margin {d[f'pred_margin_{mdl}'].corr(d.mkt_margin):.3f}")
for mdl in ["window", "ml"]:
    print(f"{mdl} total: mean |total - mkt| {np.abs(d[f'pred_total_{mdl}'] - d.mkt_total).mean():.2f}")
```
</details>

Run as `uv run python <script>.py <scratch>/nba_p3.db <scratch>` (from the repo root; `p3_wf.py` must run before `p3_move.py`, `p3_calib.py` and `p3_close.py`, which read its `wf_window_*.pkl`).
