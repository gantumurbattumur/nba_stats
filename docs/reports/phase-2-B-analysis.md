# Phase 2 — Agent B data analysis

Run 2026-11-15 on branch `dev` (HEAD `90666f7`, plus C's uncommitted ESPN 404 fix in the working tree, which doesn't touch anything analysed here). Everything ran on a private copy: `sqlite3 data/nba.db ".backup <scratch>/agentB/nba_b.db"`, then `NBA_DB_PATH=<abs>/nba_b.db`. Wider-window tables were built on further copies (`nba_b2.db` = windows 3,5,8,10,15,20; `nba_b3.db` = 20,30,40,60). SQL runs as `sqlite3 -header -column nba_b.db "<sql>"`. Python runs as `uv run python <script>`. The scripts are reproduced in the appendix (they live in the session scratchpad `agentB/`, which may be cleaned up).

Data: 3 full regular seasons (2023-24..2025-26, 1230 games each), their play-in and playoffs, and 2026-27 through 2026-11-14 (188 final games). Closing lines: 2025-26 regular season from 2025-11-24 (979 games), play-in 6/6, playoffs 25/85, 2026-27 188/188. The market sections therefore cover **1166 regular-season games** with both a spread and a total.

**This is analysis, not betting advice.** Cover/over rates describe how lines and results lined up.

## 1. Verifying A's numbers

| A's claim | Re-derived | Holds? | How |
|---|---|---|---|
| `build_feature_table` 4156 rows × 493 cols, ~0.6 s | **4156 × 496**, 0.58 / 0.55 / 0.56 s (3 runs, enriched table warm) | Rows and time yes; **cols stale** | 496 − 493 = `h_elo, a_elo, elo_prob`, joined by the later Elo commit `e1d0ce7`. A's count predates it. 4156 = 3690 regular + 18 play-in + 251 playoff + 197 (2026-27 incl. 9 scheduled). |
| `corr(d_all_l10_net, margin)` = 0.388 | **0.3885** (n=4132, all final games, all types). Regular only: 0.3978 | Yes | Independent recompute from raw `team_games` (`indep.py`, no `nba_predict` imports) matches `game_features` element-wise, max \|diff\| 5e-14, 0 NaN mismatches |
| `corr(d_all_szn_adjnet, margin)` = 0.423 | **0.4230** (n=3771). Regular only: 0.4296 | Yes | Same, max \|diff\| 5e-14. Note: adjnet is NaN until the team's games include an opponent with ≥5 games played |
| Elo log loss (2024-25 on) 0.6188 vs always-home 0.6884, acc 0.66 | **0.6188**, base-rate constant **0.6884** (home rate 0.5486), acc **0.660**, n=2829 (all types). Regular only: 0.6157 / 0.6886 / 0.663 | Yes | Independent Elo (`elo_indep.py`). `elo_prob` equals my probs exactly (max \|diff\| 0.0). p=0.5 gives 0.6931 |
| Window model 2023-25 → 2025-26: ll 0.6082, MAE 11.66, total MAE 15.30, hca 1.04, w l15 .51 / szn .46 / l5 .02 / l10 ≈0, sigma 14.5 | **0.6082, 11.66, 15.30, hca 1.04, l15 0.514 / szn 0.465 / l5 0.021 / l10 0.000, sigma 14.52** (sigma_total 18.64, b2b 2.52, rest 0.29, split_w 0.264). Test acc 0.666 | Yes (train and test include play-in and playoffs). Regular-only fit and test: 0.6057 / 11.59 / 15.27 / hca 0.99 | `wm.py`, using A's `fit`/`predict_rows`, with metrics computed by my own code |
| `nba explore --team MIL --venue away --last-n 7 --season 2026-27 --metrics record,net,ats_record,ou_record` → gp 6, 0-6, net −13.6, ATS 3-3-0, O/U 2-4-0 | CLI: `6  0-6 -13.613122  3-3-0  2-4-0`. Hand SQL below: `6  0-6  -13.6  3-3-0  2-4-0` | Yes | MIL has only 6 road games so far, so last-7 = all 6 |

```sql
with mil as (
  select t.game_id, t.pts, o.pts opp,
         (t.fga+0.44*t.fta-t.oreb+t.tov + o.fga+0.44*o.fta-o.oreb+o.tov)/2.0 gposs,
         s.total, -s.home_spread team_spread           -- MIL is away, so its spread = -home_spread
  from team_games t join team_games o on o.game_id=t.game_id and o.team_id=t.opp_team_id
  join games g on g.game_id=t.game_id
  left join odds_snapshots s on s.game_id=t.game_id and s.source='espn' and s.is_closing=1
  where t.team_id=1610612749 and t.is_home=0 and t.season='2026-27' and g.status='final'
  order by t.game_date desc limit 7)
select count(*) gp, sum(pts>opp)||'-'||sum(pts<opp) record,
  round(100.0*(sum(pts)-sum(opp))/sum(gposs),1) net,
  sum(pts-opp+team_spread>0)||'-'||sum(pts-opp+team_spread<0)||'-'||sum(pts-opp+team_spread=0) ats,
  sum(pts+opp>total)||'-'||sum(pts+opp<total)||'-'||sum(pts+opp=total) ou
from mil;
```

**Leakage spot check, passed.** My independent features use only games with `game_date < D` and the documented cross-season rule (an L-window reaches into last season while the team has fewer than 15 in-season games). They match `game_features` to 1e-13 on every game, so nothing from game day leaks in. Elo is pre-game by construction, and its probabilities match the independent Elo exactly.

**`readonly_query` guards, passed:** `delete from games`, `select 1; drop table games`, `attach database …` and `pragma writable_schema=1` all raise `QueryError`. Custom features work: `add_feature(c,'rest_adv','h_rest - a_rest')` and `'form_gap' = d_all_l15_adjnet - d_all_szn_adjnet`, then `evaluate(db)` returns 4156 × 3. On regular-season final games, corr with margin is 0.054 (rest_adv) and 0.101 (form_gap).

## 2. Which recent-form window predicts best?

Regular season only. Feature = home − away difference `d_{scope}_{window}_{net|adjnet}` (adjnet = net + strength of schedule). Corr = Pearson with the home margin over 2023-24..2026-27. OOS = OLS `margin ~ 1 + x` fit on 2023-24 + 2024-25 and scored by MAE on 2025-26. *Common* = the 1112 test games where every candidate is non-NaN (a fair head-to-head). On that sample, predicting the training mean margin alone gives **MAE 13.27**. Script: `q1.py`, on `nba_b2.db` built with `build_feature_table(conn, windows=(3,5,8,10,15,20))` (0.85 s, 684 cols).

**All-games scope** (corr / OOS MAE, common sample):

| window | net corr | net MAE | adjnet corr | adjnet MAE |
|---|---|---|---|---|
| L3 | 0.295 | 12.42 | 0.303 | 12.36 |
| L5 | 0.344 | 12.16 | 0.352 | 12.12 |
| L8 | 0.382 | 11.84 | 0.393 | 11.89 |
| L10 | 0.398 | 11.75 | 0.407 | 11.79 |
| L15 | 0.411 | 11.66 | 0.422 | 11.66 |
| **L20** | **0.421** | 11.62 | **0.432** | **11.57** |
| season | 0.405 | **11.51** | 0.430 | 11.60 |

**Split scope** (home team's home games vs away team's road games):

| window | net corr | net MAE | adjnet corr | adjnet MAE |
|---|---|---|---|---|
| L3 | 0.296 | 12.42 | 0.308 | 12.38 |
| L5 | 0.337 | 12.22 | 0.352 | 12.22 |
| L10 | 0.374 | 12.03 | 0.392 | 12.00 |
| L15 | 0.383 | 11.88 | 0.399 | 11.89 |
| L20 | 0.391 | 11.81 | 0.404 | 11.84 |
| season | 0.383 | 11.79 | 0.401 | 11.98 |

Beyond L20 (`nba_b3.db`, windows 20/30/40/60, adjnet, all scope, 1135 common 2025-26 test games): L20 0.4466 / 11.561, **L30 0.4482 / 11.560**, L40 0.4480 / 11.573, L60 0.4397 / 11.633, season 0.4440 / 11.618.

Combinations (common sample, adjnet): all 7 windows together give 11.485, with coefficients concentrated on L20 (0.38) and season (0.41). L15 + season gives 11.528. Joint all + split for the same window: L20 11.523, L15 11.603, season 11.641.

Findings:
- **Longer is better, up to about 20–30 games.** Correlation rises steadily from L3 (0.30) to L20/L30 (0.43–0.45), then flattens and drops by L60. Short windows (L3–L5) lose about 0.5–0.8 MAE against L20. Within each window, the difference between the best and worst variant is small next to the difference between windows.
- **Opponent adjustment helps a little:** adjnet beats net by about +0.01 corr at every window, and is about equal on MAE.
- **Split-only form is worse than all-games form** at every window from L8 up (by about 0.02–0.03 corr and 0.2–0.3 MAE; at L3/L5 they're about equal), because it uses half the games. Adding split on top of all-games form buys at most 0.05 MAE (L20 joint 11.523 vs 11.571).
- **Implication for A's model:** it has no L20 or L30 window, and the fit sends L10 to 0 and L5 to 0.02, the same "short windows add nothing" result. Adding L20 (and dropping L5/L10) should help slightly.

## 3. Home vs away

**By season** (`nba explore --venue home --group-by season,season_type --metrics record,win_pct,margin,ats_record,ou_record`):

| season | regular W-L | win % | margin | playoffs win % | ATS (home) | O/U |
|---|---|---|---|---|---|---|
| 2023-24 | 668-562 | 54.3 | +2.15 | 58.5 (n=82) | — | — |
| 2024-25 | 669-561 | 54.4 | +1.63 | 57.1 (n=84) | — | — |
| 2025-26 | 682-548 | 55.4 | +1.74 | 55.3 (n=85) | 497-481-0 | 473-505-0 |
| 2026-27 | 99-89 | 52.7 | +1.83 | — | 92-96-0 | 87-101-0 |

**By team** (3 full regular seasons pooled, team HCA = (home margin − road margin)/2):

```sql
with x as (select team_id, is_home, avg(plus_minus) m, avg(wl='W') w from team_games
           where season_type='regular' and season in ('2023-24','2024-25','2025-26') group by 1,2)
select team_id, round(100*h.w,1), round(h.m,2), round(100*a.w,1), round(a.m,2), round((h.m-a.m)/2,2) team_hca
from x h join x a on a.team_id=h.team_id and a.is_home=0 where h.is_home=1 order by team_hca desc;
```

The top teams: UTA 3.44, LAL 3.26, MIL 3.13, ORL 2.91, HOU 2.90. The bottom: GSW −0.38, PHI 0.22, WAS 0.31, CHI 0.70, DET 1.04 (full table in the query output). OKC is +13.2 at home and +7.7 on the road, the best in both.

**Team HCA is noise.** Correlating each team's HCA from season to season gives 2023-24 vs 2024-25 **−0.04**, 2024-25 vs 2025-26 **+0.10** and 2023-24 vs 2025-26 **−0.15**. The spread across teams (1.3–2.1 per season) is the size expected from pure sampling noise (≈ 0.5·√(2·13.5²/41) = 1.5). A single league-wide HCA is the right model. A per-team HCA in the dashboard needs a "not significant" caveat or a confidence interval.

**Does the split blend absorb home advantage?** Yes, and it fully explains the low fitted hca (`q2.py`).
- (a) The split difference is systematically higher than the all-games difference. `mean(d_split_X_net − d_all_X_net)` = **+3.88 / +3.88 / +3.95 / +3.97** per 100 possessions for L5 / L10 / L15 / season (regular season, n≈3800). The home team's home-only rating carries home court, and the away team's road-only rating carries the road penalty.
- (b) At the fitted split weight 0.264 and pace ≈ 102, the blend therefore adds 0.264 × 3.9 × 1.02 ≈ **1.05 points** of implicit home edge. Fitted hca 1.04 + 1.05 = **2.09**. That matches the hca the model fits when the split features are removed:

| variant (train 2023-25 all types, test 2025-26) | fitted hca | split_w | test MAE | test ll | mean pred vs actual margin |
|---|---|---|---|---|---|
| default (split weight free) | **1.04** | 0.264 | 11.657 | 0.6082 | +2.00 vs +1.77 |
| no split (split cols set to NaN) | **2.18** | — | 11.681 | 0.6075 | +2.16 vs +1.77 |
| split only (all-scope := split) | **−2.14** | — | 12.030 | 0.6275 | +1.51 vs +1.77 |

- So the model's real home edge is about 2.1 points, consistent with the observed +1.6 to +2.2. Split form adds only about 0.02 MAE, and the version without split has slightly *better* log loss.
- In the single-feature OLS of §2, all-scope fits have an intercept of about +1.9–2.0, while split fits have −0.2 to −1.2. Same effect.

## 4. Rest and schedule

Team-perspective rows from `team_games_enriched`, regular season 2023-24..2026-27. ATS only where a line exists (2025-26 from 11-24, plus 2026-27). `rest_days` = days off before the game, capped at 7. Season openers get 7 by default, so I split them out:

```sql
with e as (select *, lag(game_date) over (partition by team_id, season order by game_date) prev
           from team_games_enriched where season_type='regular')
select case when prev is null then 'opener' when rest_days>=3 then '3+' else cast(rest_days as text) end rest,
  count(*) n, round(100.0*avg(win),1) win_pct, round(avg(margin),2) margin,
  round(100.0*sum(ats='cover')/nullif(sum(ats in ('cover','no_cover')),0),1) ats_pct,
  sum(ats in ('cover','no_cover')) n_ats
from e group by 1 order by 1;
```

| rest days | n | win % | margin | ATS % (n) |
|---|---|---|---|---|
| 0 (back-to-back) | 1379 | 45.9 | −1.71 | 51.8 (421) |
| 1 | 4799 | 50.5 | +0.25 | 49.4 (1433) |
| 2 | 1085 | 53.1 | +0.90 | 50.6 (332) |
| 3+ | 373 | 50.7 | +0.58 | 50.0 (116) |
| opener | 120 | 47.5 | −0.45 | 46.7 (30) |

Back-to-back × 3-in-4 (same table, `group by b2b, three_in_four`):

| b2b | 3-in-4 | n | win % | margin | ATS % (n) |
|---|---|---|---|---|---|
| no | no | 5344 | 51.3 | +0.51 | 50.1 (1586) |
| no | yes | 1033 | 49.0 | −0.34 | 47.4 (325) |
| yes | no | 381 | 44.9 | −1.65 | 49.1 (112) |
| yes | yes | 998 | 46.3 | −1.73 | 52.8 (309) |

By venue: home on b2b 48.7% / −0.09 (rested home 55.9% / +2.24), and away on b2b 43.2% / −3.26 (rested away 45.9% / −1.52). A back-to-back costs about 1.7–2.3 points, close to the model's fitted `b2b_pts` = 2.5.

**Rest advantage** (game level, from `game_features`, home rest − away rest):

```sql
select case when h_rest-a_rest<=-2 then '<=-2' when h_rest-a_rest>=2 then '>=+2' else h_rest-a_rest end adv,
  count(*), round(100*avg(home_win),1), round(avg(margin),2),
  round(avg(case when m_home_spread is not null then -m_home_spread end),2) mkt_expected,
  round(100.0*sum(margin+m_home_spread>0)/nullif(sum(m_home_spread is not null and margin+m_home_spread<>0),0),1) home_cover
from game_features where status='final' and season_type='regular' and h_gp_szn>0 and a_gp_szn>0 group by 1;
```

| home − away rest | n | home win % | home margin | market expected margin | home cover % (n lines) |
|---|---|---|---|---|---|
| ≤ −2 | 154 | 51.3 | +0.44 | +0.45 | 55.8 (43) |
| −1 | 656 | 50.0 | −0.12 | +0.33 | 52.3 (199) |
| 0 | 2104 | 54.8 | +1.96 | +1.85 | 49.4 (633) |
| +1 | 725 | 57.2 | +2.94 | +2.87 | 48.5 (227) |
| ≥ +2 | 176 | 63.1 | +4.56 | +2.94 | 58.3 (48) |

Rest is a real, monotone effect on results (about 1 point per day of advantage), and **the market already prices it**: the expected margin tracks the actual one in every bucket with a decent sample. The outer buckets with n=43–48 lines are noise (1 SE ≈ 7.5 pts). Head-to-head b2b grid (both rested / only away b2b / only home b2b / both b2b): home margin +1.97 / +3.73 / −1.10 / +2.15, against market expectations of +1.84 / +3.84 / +0.23 / +0.36.

## 5. Market (2025-26 + 2026-27, regular season, 1165–1166 games with a line)

Game level (one row per game; the favourite is the negative-spread side, and there are no pick'ems), `q4a.py`. 1 SE ≈ 5.0 / 2.9 / 1.6 pts at n = 100 / 300 / 1000.

| split | n | fav cover % | home cover % | over % |
|---|---|---|---|---|
| all, 2025-26 | 977 | 49.4 | 50.9 | 48.3 |
| all, 2026-27 | 188 | 53.7 | 48.9 | 46.3 |
| **all, both** | 1165 | 50.1 | 50.6 | 48.0 |
| home favourite | 680 | 50.6 | 50.6 | 47.8 |
| road favourite | 485 | 49.5 | 50.5 | 48.2 |
| \|spread\| 1–3 | 241 | 47.3 | 49.0 | **41.5** |
| \|spread\| 3.5–6.5 | 404 | 53.5 | 51.7 | 49.0 |
| \|spread\| 7–9.5 | 219 | 47.5 | 50.7 | 51.1 |
| \|spread\| 10+ | 301 | 49.8 | 50.2 | 49.5 |
| total ≤ 220 | 127 | 44.9 | 44.9 | 47.2 |
| total 220.5–230 | 458 | 52.8 | 52.4 | 46.9 |
| total 230.5–240 | 450 | 51.3 | 51.1 | 49.3 |
| total > 240 | 130 | 41.5 | 47.7 | 47.7 |
| neither team b2b | 818 | 49.8 | 50.5 | 46.2 |
| only home b2b | 135 | 45.9 | 54.1 | 52.6 |
| only away b2b | 139 | 50.4 | 48.9 | 51.1 |
| both b2b | 73 | 61.6 | 47.9 | 53.4 |

Team-perspective cover % for a team on a back-to-back: as favourite 55.1% (n=187), as underdog 49.1% (234). Rested: favourite 49.1% (979), underdog 50.1% (932).

**Read:** every cell is within about 2 SE of 50%. The largest deviation is the over in close games (\|spread\| ≤ 3: 41.5%, about −2.6 SE), but with about 40 cells tested, one result that size is expected by chance. Overs are slightly under 50% overall (48.0%, about −1.4 SE). Season-level splits are shown in the script output (for example, 2026-27 \|spread\| 10+ overs 32.4% on n=37).

**Window model vs the closing spread** (`q4b.py`). Out of sample: 2025-26 predictions come from the model fit on 2023-25, and 2026-27 predictions from a refit on 2023-26. MAE of the home margin, with the market prediction = `−home_spread`. `ols_l20` = `0.755·d_all_l20_adjnet + 1.94` (fit on 2023-25 regular). `avg` = mean of model and market.

| subset | n | market | window model | OLS L20 | avg(model, mkt) | corr(model, mkt) | total: market | total: model |
|---|---|---|---|---|---|---|---|---|
| 2025-26 | 978 | **11.06** | 11.72 | 11.74 | 11.26 | 0.88 | **14.68** | 15.40 |
| 2026-27 | 188 | **11.26** | 12.94 | 12.31 | 11.76 | 0.80 | **14.03** | 14.55 |
| either team < 15 gp | 189 | **11.28** | 12.96 | 12.36 | 11.78 | 0.80 | 14.05 | 14.60 |
| both ≥ 15 gp | 977 | **11.06** | 11.71 | 11.73 | 11.25 | 0.88 | 14.67 | 15.39 |
| \|spread\| 1–3 | 242 | **12.48** | 12.98 | 12.94 | 12.64 | 0.28 | 14.78 | 15.24 |
| \|spread\| 3.5–6.5 | 404 | **11.33** | 12.36 | 12.20 | 11.69 | 0.72 | 14.42 | 15.22 |
| \|spread\| 7–9.5 | 219 | **9.51** | 10.51 | 10.26 | 9.84 | 0.84 | 14.62 | 15.61 |
| \|spread\| 10+ | 301 | **10.82** | 11.47 | 11.57 | 10.90 | 0.93 | 14.59 | 15.08 |
| any b2b | 348 | **11.31** | 12.26 | 12.25 | 11.66 | 0.86 | 15.25 | 15.90 |
| no b2b | 818 | **11.00** | 11.77 | 11.65 | 11.20 | 0.85 | 14.29 | 14.99 |
| month 2025-11 (from 11-24) | 47 | 9.78 | 10.59 | 9.83 | 10.01 | 0.90 | 16.44 | 17.23 |
| month 2026-03 | 239 | **10.69** | 11.09 | 11.25 | 10.77 | 0.93 | 13.86 | 14.73 |
| month 2026-10 | 88 | **11.34** | 13.74 | 13.09 | 12.16 | 0.74 | 14.78 | 15.41 |

**There is no window, split or subset where the model's margin beats the closing spread.** That holds for every month, spread bucket, rest state and early/late season. The closest is March 2026 (10.69 vs 11.09). Averaging the model with the market is also worse than the market alone. Both checks below agree:
- **Blend:** fit `margin ~ 1 + mkt + (model − mkt)` on 2025-26 and get `−0.32 + 1.014·mkt − 0.168·(model − mkt)`. The model's disagreement gets a *negative* weight. Out of sample on 2026-27: market 11.261, blend 11.361.
- **Disagreement buckets** (model − market, + = model likes home more; both seasons, n=1166). Home cover % by bucket: ≤−4 49.5%, (−4,−2] 51.6%, (−2,0] 56.5%, (0,2] 50.5%, (2,4] 52.9%, **>4 43.0% (n=214, mean margin − market = −3.05)**. `corr(gap, margin − mkt)` = −0.05. When the model is far from the line, the line has been closer to the result.

Early-season form is the model's weakest point. In October 2026 its MAE is 13.74 against the market's 11.34, and its correlation with the market drops to 0.74. The cross-season L-windows carry last season's teams into a new season with new rosters, while the market knows about the roster changes.

## 6. Anything surprising

1. **The fitted hca of about 1 is an artefact, not a finding.** The split blend carries about 1.05 points of home edge, and the true total is about 2.1 (§3). Dashboard copy like "home court is worth 1 point" would be wrong.
2. **Short windows are the weakest predictors** (L3/L5 ≈ 0.30–0.35 corr against 0.43–0.45 for L20–L30), yet they're what a "recent form" view naturally shows. "Last 5" is mostly noise for predicting the next game's margin.
3. **Per-team home advantage doesn't repeat** from season to season (r ≈ 0), so rankings like "UTA has the best home court" are noise.
4. **The market prices rest well.** Results move about 1 point per rest day, and closing spreads move with them.
5. **The model's large disagreements with the market point the wrong way** (home covers 43% when the model is more than 4 points higher on the home team). Any future value-flag feature should show this calibration table next to each flag.
6. **Housekeeping:**
   - `team_games_enriched.rest_days` gives season openers 7, which mixes them into the 3+ bucket. There are only 120 openers, but they should be flagged.
   - `build_feature_table`'s columns grow with every window: 684 cols for 6 windows. That's fine at 0.85 s.

## Next

- **Feature (model, A): add L20 (or L30) and drop L5/L10 from the default window set** (`WindowParams.weights` and `build_feature_table` windows). On 2025-26 OOS, the single-feature L20 adjnet is the best all-games window (MAE 11.57), and the fit already zeroes L10 and nearly zeroes L5.
- **Fix (model, A): make HCA identifiable.** Either center the split features by subtracting the league-wide home/road split gap (≈ 3.9 net/100), so `hca` means what it says, or report "effective HCA = hca + split_w·gap·pace/100" in the model card and dashboard. The current `hca=1.04` is misleading. Test: with split columns NaN'd, the fit returns hca ≈ 2.2.
- **Feature (dashboard): a "Model vs market" view.**
  - Rolling 50-game MAE of model and closing spread by date.
  - The model-minus-market gap calibration table from §5 (home cover % by gap bucket).
  - The same split by early season (< 15 gp).
  - This makes the model's honest standing visible: about 0.65 MAE behind the close overall, about 2.4 behind in the first month.
- **Feature (derived): an early-season prior.** For the first ~15 games, shrink the cross-season L-windows toward a regressed prior-season rating (or the Elo margin `elo_margin`, which is already in `run_elo` but not joined to `game_features`). October 2026 is where the model is worst.
- **Feature (dashboard): show the window-strength chart from §2** (corr by window, all vs split, net vs adjnet) next to the form widgets, so users see that L3/L5 are weak predictors.
- **Small data fix:** flag season openers (`is_opener`) in `team_games_enriched` instead of `rest_days = 7`.

## Appendix: scripts

<details><summary><code>indep.py</code>: independent window features (args: db, windows, out)</summary>

```python
import sqlite3, sys, numpy as np, pandas as pd
DB = sys.argv[1]
db = sqlite3.connect(DB)
t = pd.read_sql("""select t.game_id, t.team_id, t.opp_team_id, t.season, t.season_type, t.game_date, t.is_home, t.pts,
  o.pts opp_pts, (t.fga+0.44*t.fta-t.oreb+t.tov + o.fga+0.44*o.fta-o.oreb+o.tov)/2.0 gposs
  from team_games t join team_games o on o.game_id=t.game_id and o.team_id=t.opp_team_id
  join games g on g.game_id=t.game_id where g.status='final'""", db)
t = t.sort_values(["team_id","game_date"]).reset_index(drop=True)
def szn_state(df):  # season-to-date state strictly before each game
    out = {}
    for (tid, s), g in df.groupby(["team_id","season"]):
        cp = co = cg = n = 0
        for r in g.sort_values("game_date").itertuples():
            out[(tid, r.game_date)] = (n, cp, co, cg)
            cp += r.pts; co += r.opp_pts; cg += r.gposs; n += 1
    return out
st = szn_state(t)
def pre_net(tid, d):
    n, cp, co, cg = st[(tid, d)]
    return 100*(cp-co)/cg if n >= 5 else np.nan
t["opp_net_pre"] = [pre_net(o, d) for o, d in zip(t.opp_team_id, t.game_date)]
def feats_before(g, season, date, wins):
    prior = g[g.game_date < date]; ins = prior[prior.season == season]; out = {}
    for w in wins:
        sel = ins if w == "szn" else (prior.tail(w) if len(ins) < 15 else ins.tail(w))
        if len(sel) == 0:
            out[f"{w}_net"] = out[f"{w}_adjnet"] = out[f"{w}_adjnet_strict"] = np.nan; out[f"{w}_gp"] = 0; continue
        net = 100*(sel.pts.sum()-sel.opp_pts.sum())/sel.gposs.sum(); sos = sel.opp_net_pre.dropna()
        out[f"{w}_net"] = net
        out[f"{w}_adjnet"] = net + (sos.mean() if len(sos) else 0.0)
        out[f"{w}_adjnet_strict"] = net + sos.mean() if len(sos) else np.nan   # = A's definition
        out[f"{w}_gp"] = len(sel)
    return out
games = pd.read_sql("select game_id, season, game_date, home_team_id, away_team_id, home_pts-away_pts margin from games where status='final'", db)
by = {tid: g for tid, g in t.groupby("team_id")}
WINS = [int(x) if x.isdigit() else x for x in sys.argv[2].split(",")]
rows = []
for r in games.itertuples():
    h = feats_before(by[r.home_team_id], r.season, r.game_date, WINS); a = feats_before(by[r.away_team_id], r.season, r.game_date, WINS)
    rows.append({"game_id": r.game_id, "margin": r.margin, **{f"d_{k}": h[k] - a[k] for k in h}})
out = pd.DataFrame(rows); out.to_parquet(sys.argv[3])
for c in out.columns:
    if c.startswith("d_") and not c.endswith("_gp"): print(c, round(out[c].corr(out.margin), 4), out[c].notna().sum())
```
Run: `uv run python indep.py nba_b.db 10,szn indep.parquet`. Then compare it with `game_features.d_all_l10_net` / `d_all_szn_adjnet` by merging on `game_id`, which gives max \|diff\| 5e-14.
</details>

<details><summary><code>elo_indep.py</code></summary>

```python
import sqlite3, sys, numpy as np, pandas as pd
db = sqlite3.connect(sys.argv[1])
g = pd.read_sql("select game_id, season, season_type, game_date, home_team_id h, away_team_id a, home_pts hp, away_pts ap, status from games order by game_date, game_id", db)
R, last, probs = {}, {}, []
for r in g.itertuples():
    for t in (r.h, r.a):
        if t not in R: R[t] = 1505.0
        elif last[t] != r.season: R[t] = 1505 + 0.75 * (R[t] - 1505)
        last[t] = r.season
    d = R[r.h] - R[r.a] + 100; p = 1 / (1 + 10 ** (-d / 400)); probs.append(p)
    if r.status != "final": continue
    mov = r.hp - r.ap; hw = mov > 0
    mult = (abs(mov) + 3) ** 0.8 / (7.5 + 0.006 * (d if hw else -d))
    delta = 20 * mult * ((1.0 if hw else 0.0) - p); R[r.h] += delta; R[r.a] -= delta
g["p"] = probs; g["y"] = (g.hp > g.ap).astype(float)
f = g[(g.status == "final") & (g.season >= "2024-25")]
ll = lambda p, y: float(-np.mean(y*np.log(np.clip(p,1e-12,1-1e-12)) + (1-y)*np.log(np.clip(1-p,1e-12,1))))
for name, sub in [("all types", f), ("regular", f[f.season_type == "regular"])]:
    print(name, len(sub), ll(sub.p, sub.y), ((sub.p > .5) == (sub.y == 1)).mean(), ll(np.full(len(sub), sub.y.mean()), sub.y))
```
</details>

<details><summary><code>wm.py</code>: window model check (A's fit, my metrics)</summary>

```python
import sqlite3, numpy as np, pandas as pd
from nba_predict.models.window_rating import fit, predict_rows
f = pd.read_sql("select * from game_features", sqlite3.connect("nba_b.db"))
ll = lambda p, y: float(-np.mean(y*np.log(np.clip(p,1e-12,1-1e-12)) + (1-y)*np.log(np.clip(1-p,1e-12,1))))
tr = f[f.season.isin(["2023-24","2024-25"]) & f.margin.notna()]; te = f[(f.season=="2025-26") & f.margin.notna()]
p = fit(tr); pr = predict_rows(te, p).merge(te[["game_id","margin","total","home_win"]], on="game_id")
print(p.hca, p.weights, p.sigma, ll(pr.home_win_prob, pr.home_win),
      (pr.pred_margin-pr.margin).abs().mean(), (pr.pred_total-pr.total).abs().mean())
# regular-only variant: add (season_type=="regular") to both masks
```
</details>

<details><summary><code>q1.py</code>: window comparison</summary>

```python
import numpy as np, pandas as pd
f = pd.read_parquet("gf_wide.parquet")   # build_feature_table(conn, windows=(3,5,8,10,15,20)) on a copy
f = f[(f.season_type == "regular") & f.margin.notna()]
W = ["l3","l5","l8","l10","l15","l20","szn"]
feats = [f"d_{sc}_{w}_{m}" for m in ("net","adjnet") for sc in ("all","split") for w in W]
tr = f[f.season.isin(["2023-24","2024-25"])]; te = f[f.season == "2025-26"]
ctr, cte = tr.dropna(subset=feats), te.dropna(subset=feats)          # common sample (1112 test games)
def ols(Xtr, ytr, Xte):
    b = np.linalg.lstsq(np.column_stack([np.ones(len(Xtr)), Xtr]), ytr, rcond=None)[0]
    return np.column_stack([np.ones(len(Xte)), Xte]) @ b, b
for c in feats:
    a = f[[c,"margin"]].dropna()
    pc, b = ols(ctr[[c]].values, ctr.margin.values, cte[[c]].values)
    print(c, a[c].corr(a.margin), np.abs(pc - cte.margin).mean(), b)
# joint: ols(ctr[[f"d_all_{w}_adjnet", f"d_split_{w}_adjnet"]], ...); all windows: ols(ctr[[f"d_all_{w}_adjnet" for w in W]], ...)
# L30/L40/L60: same, on build_feature_table(conn, windows=(20,30,40,60)), common sample over l20..l60+szn adjnet
```
</details>

<details><summary><code>q2.py</code>: HCA absorption</summary>

```python
import sqlite3, numpy as np, pandas as pd
from nba_predict.models.window_rating import fit, predict_rows
f = pd.read_sql("select * from game_features", sqlite3.connect("nba_b.db")); fin = f[f.margin.notna()]
reg = fin[fin.season_type == "regular"]
for w in ["l5","l10","l15","szn"]:
    print(w, (reg[f"d_split_{w}_net"] - reg[f"d_all_{w}_net"]).dropna().mean())
tr = fin[fin.season.isin(["2023-24","2024-25"])]; te = fin[fin.season == "2025-26"]
split = [c for c in f.columns if "_split_" in c]
def ev(trf, tef):
    p = fit(trf); pr = predict_rows(tef, p).merge(tef[["game_id","margin"]], on="game_id")
    print(p.hca, p.split_weight, np.abs(pr.pred_margin - pr.margin).mean())
ev(tr, te)                                                                                  # default
ev(tr.assign(**{c: np.nan for c in split}), te.assign(**{c: np.nan for c in split}))       # no split
def split_only(d):
    d = d.copy()
    for c in split: d[c.replace("_split_","_all_")] = d[c]
    return d
ev(split_only(tr), split_only(te))                                                          # split only
```
</details>

<details><summary><code>q4a.py</code>: ATS / O-U splits</summary>

```python
import sqlite3, numpy as np, pandas as pd
g = pd.read_sql("""select game_id, season, margin, total, m_home_spread hs, m_total tl, h_b2b, a_b2b
  from game_features where status='final' and season_type='regular' and m_home_spread is not null and m_total is not null""",
  sqlite3.connect("nba_b.db"))
g["home_fav"] = g.hs < 0
g["fav_cover"] = np.where(g.home_fav, g.margin + g.hs > 0, -(g.margin + g.hs) > 0)
g["home_cover"] = g.margin + g.hs > 0; g["over"] = g.total > g.tl
g["sb"] = pd.cut(g.hs.abs(), [0,3,6.5,9.5,30]); g["tb"] = pd.cut(g.tl, [0,220,230,240,300])
g["favv"] = np.where(g.home_fav, "home fav", "road fav")
g["b2bk"] = np.select([(g.h_b2b==1)&(g.a_b2b==0), (g.h_b2b==0)&(g.a_b2b==1), (g.h_b2b==1)&(g.a_b2b==1)],
                      ["home b2b only","away b2b only","both b2b"], "neither b2b")
for k in ["favv","sb","tb","b2bk"]:
    print(g.groupby(k, observed=True)[["fav_cover","home_cover","over"]].agg(["size","mean"]))
# team view: select b2b, is_favorite, ats from team_games_enriched where season_type='regular' and ats in ('cover','no_cover')
```
</details>

<details><summary><code>q4b.py</code>: model vs closing spread</summary>

```python
import sqlite3, numpy as np, pandas as pd
from nba_predict.models.window_rating import fit, predict_rows
f = pd.read_sql("select * from game_features", sqlite3.connect("nba_b.db")); fin = f[f.margin.notna()]
fw = pd.read_parquet("gf_wide.parquet")[["game_id","d_all_l20_adjnet"]]
pr = pd.concat([predict_rows(fin[fin.season == s], fit(fin[fin.season.isin(t)]))
                for s, t in [("2025-26", ["2023-24","2024-25"]), ("2026-27", ["2023-24","2024-25","2025-26"])]])
d = fin.merge(pr, on="game_id").merge(fw, on="game_id", how="left")
d = d[(d.season_type == "regular") & d.m_home_spread.notna()].copy()
tr = fin[fin.season.isin(["2023-24","2024-25"]) & (fin.season_type=="regular")].merge(fw, on="game_id").dropna(subset=["d_all_l20_adjnet"])
b = np.polyfit(tr.d_all_l20_adjnet, tr.margin, 1); d["ols_l20"] = np.polyval(b, d.d_all_l20_adjnet.fillna(0))
d["mkt"] = -d.m_home_spread
d["early"] = (d.h_gp_szn < 15) | (d.a_gp_szn < 15); d["month"] = d.game_date.str[:7]
d["sb"] = pd.cut(d.m_home_spread.abs(), [0,3,6.5,9.5,30]); d["anyb2b"] = (d.h_b2b == 1) | (d.a_b2b == 1)
for k in ["season","early","month","sb","anyb2b"]:
    print(d.groupby(k, observed=True).apply(lambda x: pd.Series(dict(n=len(x),
        market=np.abs(x.mkt-x.margin).mean(), model=np.abs(x.pred_margin-x.margin).mean(),
        ols=np.abs(x.ols_l20-x.margin).mean(), avg=np.abs((x.pred_margin+x.mkt)/2-x.margin).mean(),
        tot_mkt=np.abs(x.m_total-x.total).mean(), tot_model=np.abs(x.pred_total-x.total).mean()))))
a, z = d[d.season == "2025-26"], d[d.season == "2026-27"]
X = lambda x: np.column_stack([np.ones(len(x)), x.mkt, x.pred_margin - x.mkt])
coef = np.linalg.lstsq(X(a), a.margin, rcond=None)[0]; print(coef, np.abs(X(z) @ coef - z.margin).mean())
d["gap"] = d.pred_margin - d.mkt; d["home_cover"] = d.margin + d.m_home_spread > 0
print(d.groupby(pd.cut(d.gap, [-50,-4,-2,0,2,4,50]), observed=True).home_cover.agg(["size","mean"]))
```
</details>
