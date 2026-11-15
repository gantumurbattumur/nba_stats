# Phase 1 — Agent B data analysis

DB: `data/nba.db`, run 2026-11-15 on branch `dev`. All SQL runs as `sqlite3 -header -column data/nba.db "<sql>"` from the repo root unless stated otherwise. Scratch scripts are under the session scratchpad `agentB/` and are reproduced inline where a number depends on them.

**Mid-run schema change.** While the backfill ran, commit `76aeeca` (C) added `season_type` (`regular` / `playin` / `playoffs`) to `games` and `team_games` and re-ingested play-in and playoff games. Every regular-season number below filters `season_type='regular'`. Before that change the same queries without the filter gave identical regular-season results.

## 1. Backfill runs

| command | result |
|---|---|
| `uv run nba ingest --season 2023-24 --season 2024-25 --season 2025-26` | `2460 team_games / 1230 final games` for each season, no warnings |
| `uv run nba espn --from 2026-10-20 --to 2026-11-14` | `wrote 172 ESPN snapshots; final games 188, linked 188, with lines 188`, exit 0 |
| `uv run nba espn --from 2025-10-21 --to 2026-04-12` | `wrote 979 ESPN snapshots; final games 1230, linked 1230, with lines 979`, exit 0 |
| extra: `uv run nba espn --from 2026-04-13 --to 2026-06-13` (play-in + playoffs, after C's `season_type` change) | **exit 1** on 2026-04-25: ESPN summary 401869391 → HTTP 404 |
| extra resume: `--from 2026-04-26 --to 2026-06-13` | **exit 1**: 404 on 401869377 |
| extra resume: `--from 2026-04-27 --to 2026-06-13` | **exit 1**: 404 on 401869373. Stopped here; postseason is linked through 2026-04-27 only |

The three requested runs had no crashes, retries or warnings. The postseason runs (not requested, but the data now exists) all abort on ESPN 404s; see Next.

## 2. Row counts and integrity

```sql
select season, season_type, count(*) games, sum(status='final') final, min(game_date), max(game_date)
from games group by 1,2;
select season, season_type, count(*) rows, count(distinct game_id) gids from team_games group by 1,2;
```

| season | type | games | final | first | last | team_games |
|---|---|---|---|---|---|---|
| 2023-24 | regular | 1230 | 1230 | 2023-10-24 | 2024-04-14 | 2460 |
| 2023-24 | playin / playoffs | 6 / 82 | all | | 2024-06-17 | 12 / 164 |
| 2024-25 | regular | 1230 | 1230 | 2024-10-22 | 2025-04-13 | 2460 |
| 2024-25 | playin / playoffs | 6 / 84 | all | | 2025-06-22 | 12 / 168 |
| 2025-26 | regular | 1230 | 1230 | 2025-10-21 | 2026-04-12 | 2460 |
| 2025-26 | playin / playoffs | 6 / 85 | all | | 2026-06-13 | 12 / 170 |
| 2026-27 | regular | 197 | 188 (+9 `scheduled` for 2026-11-15) | 2026-10-20 | 2026-11-15 | 376 |

Every check below returned **0**:

```sql
-- final game without exactly 2 team rows
select count(*) from games g where g.status='final'
  and (select count(*) from team_games t where t.game_id=g.game_id)<>2;
-- orphan team_games
select count(*) from team_games t where not exists (select 1 from games g where g.game_id=t.game_id);
-- duplicate (date, home, away)
select count(*) from (select game_date, home_team_id, away_team_id, count(*) c from games group by 1,2,3 having c>1);
-- a team twice on one date
select count(*) from (select team_id, game_date, count(*) c from team_games group by 1,2 having c>1);
-- games vs team_games: points, home/away ids, opp ids
select count(*) from games g join team_games h on h.game_id=g.game_id and h.is_home=1
  join team_games a on a.game_id=g.game_id and a.is_home=0
 where h.pts<>g.home_pts or a.pts<>g.away_pts or h.team_id<>g.home_team_id or a.team_id<>g.away_team_id
    or h.opp_team_id<>a.team_id or a.opp_team_id<>h.team_id;
-- exactly one home row per game
select count(*) from (select game_id, sum(is_home) s from team_games group by game_id having s<>1);
-- W/L agrees with points; ties
select count(*) from team_games t join team_games o on o.game_id=t.game_id and o.team_id=t.opp_team_id
 where not ((t.pts>o.pts and t.wl='W') or (t.pts<o.pts and t.wl='L'));
select count(*) from games where home_pts=away_pts;
-- plus_minus = pts - opp pts
select count(*) from team_games t join team_games o on o.game_id=t.game_id and o.team_id=t.opp_team_id
 where t.plus_minus <> t.pts-o.pts;
-- season label agrees with date (Aug-Dec -> YYYY-YY+1)
select count(*) from team_games t join games g using(game_id) where t.game_date<>g.game_date or t.season<>g.season;
-- nulls in min/pts/fga/fta/oreb/tov/plus_minus
select season, sum(min is null)+sum(pts is null)+sum(fga is null)+sum(fta is null)+sum(oreb is null)
              +sum(tov is null)+sum(plus_minus is null) from team_games group by season;
```

Per-team regular-season games: 30 teams, exactly 82 games each in all three full seasons (11–14 so far in 2026-27).

```sql
select season, count(distinct team_id), min(n), max(n) from
 (select season, team_id, count(*) n from team_games where season_type='regular' group by 1,2) group by season;
select season, min(hg), max(hg) from
 (select season, team_id, sum(is_home) hg from team_games where season_type='regular' group by 1,2) group by season;
```

Home games per team range **40–42**, not exactly 41. This is real (NBA Cup knockout hosting plus the international games), not a bug, but anything that assumes 41/41 is wrong.

## 3. Home advantage

```sql
select season, count(*) n, round(100.0*avg(home_pts>away_pts),1) home_win_pct,
       round(avg(home_pts-away_pts),2) mean_home_margin
from games where status='final' and season_type='regular' group by season;
```

| season | n | home win % | mean home margin |
|---|---|---|---|
| 2023-24 | 1230 | 54.3 | +2.15 |
| 2024-25 | 1230 | 54.4 | +1.63 |
| 2025-26 | 1230 | 55.4 | +1.74 |
| 2026-27 | 188 | 52.7 | +1.83 |

All inside the expected 54–58% / +1.5–3.5 band, except 2026-27, which is low on win % at n=188 (one standard error is about 3.6 pts of win %, so this is noise). Excluding the 14 neutral-site games (list in §6) moves no season by more than 0.2 pts of win %.

Postseason, for reference (same query with `season_type<>'regular'`, grouped by type): playoffs 58.5 / 57.1 / 55.3% home win (n=82/84/85), play-in n=6 per season. Playoff ATS on the 25 lined 2025-26 games: home covered 64.0%, over 36.0%, mean |margin+spread| 10.26 vs 15.14 if flipped (sign OK, n is small).

## 4. Points, possessions, pace

Possessions use the spec's estimate `FGA + 0.44·FTA − OREB + TOV`, averaged over the two teams. Pace uses game minutes = the larger of the two teams' `min` (see the DEN glitch in §7).

```sql
with p as (select season, game_id, pts, fga+0.44*fta-oreb+tov poss, min from team_games where season_type='regular'),
     g as (select season, game_id, avg(poss) gposs, max(min) gmin, avg(pts) pts from p group by 1,2)
select season, round(avg(pts),1) mean_team_pts, round(avg(gposs),1) mean_poss,
       round(avg(gposs*48.0/(gmin/5.0)),1) pace_per48, round(100*avg(pts/gposs),1) ortg
from g group by season;
```

| season | mean team pts | mean poss | pace /48 | ORtg (pts/100 poss) |
|---|---|---|---|---|
| 2023-24 | 114.2 | 101.5 | 100.9 | 112.5 |
| 2024-25 | 113.8 | 102.0 | 101.4 | 111.7 |
| 2025-26 | 115.6 | 102.6 | 102.0 | 112.8 |
| 2026-27 | 113.6 | 102.5 | 101.9 | 110.8 |

Plausible and inside 98–102. Two caveats:
- The simple estimate runs about 2 possessions high against the NBA's published pace (2023-24 official ≈ 98.5–99), so ORtg is about 2–3 points low (official 2023-24 ≈ 115.3). That's fine for relative comparisons between teams, but should be labelled "estimated" in the dashboard.
- The two teams' possession estimates differ by 2.3 on average (max 13.1), so per-team possessions are noisy. Using the game average for both teams (as above) is the better choice for ratings.

```sql
with p as (select game_id, season, is_home, fga+0.44*fta-oreb+tov poss from team_games)
select h.season, round(avg(abs(h.poss-a.poss)),2), round(max(abs(h.poss-a.poss)),1)
from p h join p a on a.game_id=h.game_id and a.is_home=0 where h.is_home=1 group by h.season;
```

Score ranges: team points from 66 to 162, game totals from 152 to 309. No zero or impossible scores. Overtime rows (`min` ≥ 265): 118 / 120 / 108 per full season.

## 5. ESPN line coverage

```sql
select g.season, g.season_type, count(*) final_games, sum(g.espn_event_id is not null) linked,
  sum(exists(select 1 from odds_snapshots o where o.game_id=g.game_id and o.source='espn')) with_line,
  round(100.0*avg(exists(select 1 from odds_snapshots o where o.game_id=g.game_id and o.source='espn')),1) pct
from games g where g.status='final' group by 1,2;
```
(`uv run nba status` prints the same linked/lines columns and agrees.)

| season | type | final | linked | with line | % |
|---|---|---|---|---|---|
| 2023-24 | regular | 1230 | 0 | 0 | 0.0 (not backfilled; ESPN has none per spec) |
| 2024-25 | regular | 1230 | 0 | 0 | 0.0 (same) |
| **2025-26** | **regular** | **1230** | **1230** | **979** | **79.6** |
| 2025-26 | playin | 6 | 6 | 6 | 100 |
| 2025-26 | playoffs | 85 | 25 | 25 | 29.4 (blocked by ESPN 404s, see Next) |
| **2026-27** | **regular** | **188** | **188** | **188** | **100.0** |

**The 2025-26 gap is at the source, not in the parser.** Coverage by month:

```sql
select substr(game_date,1,7) m, count(*) games, sum(o.game_id is not null) lines
from games g left join odds_snapshots o on o.game_id=g.game_id and o.source='espn'
where g.season='2025-26' and g.season_type='regular' group by 1;
```

| month | 2025-10 | 2025-11 | 2025-12 | 2026-01 | 2026-02 | 2026-03 | 2026-04 |
|---|---|---|---|---|---|---|---|
| games / lines | 80 / 0 | 219 / 48 | 197 / 197 | 233 / 233 | 166 / 166 | 239 / 239 | 96 / 96 |

- The first line is on **2025-11-24** (`select min(g.game_date) from games g join odds_snapshots o using(game_id) where g.season='2025-26'`). From then on, coverage is 100% except game `0022500283` (DET @ IND, 2025-11-24, event 401810138).
- The raw summary for an early game (event 401810061, ATL @ LAC 2025-11-10) has `"pickcenter": []` and `"odds": []`. A late one (401809811) has a full Draft Kings entry. So the 250 missing games from opening night to 2025-11-23 are an **ESPN gap**, and the spec's "DraftKings for 2025-26" should read "from 2025-11-24".
- The same game is available from ESPN's core API: `sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/401810061/competitions/401810061/odds` returns provider **"ESPN BET"**, spread −4.5, total 221.5, ML −185/+155, with `open`/`close` sub-objects. The sign is home-perspective (`details: "LAC -4.5"`, LAC home). The items list also includes an "ESPN Bet - Live Odds" entry, which must be skipped. See Next.

**Books:** only `Draft Kings` (source `espn`), one snapshot per game, `is_closing=1`.
```sql
select g.season, o.source, o.book, count(*), count(distinct o.game_id) from odds_snapshots o join games g using(game_id) group by 1,2,3;
select count(*) from (select game_id from odds_snapshots where source='espn' group by 1 having count(*)>1);   -- 0
select count(*) from odds_snapshots o where not exists (select 1 from games g where g.game_id=o.game_id);     -- 0 orphans
select count(*) from (select espn_event_id from games where espn_event_id is not null group by 1 having count(*)>1); -- 0
```

**Null-field rates** (% of espn snapshots, regular season):
```sql
select g.season, count(*), round(100.0*avg(home_ml is null),1), round(100.0*avg(home_spread is null),1),
  round(100.0*avg(home_spread_price is null),1), round(100.0*avg(total is null),1), round(100.0*avg(over_price is null),1)
from odds_snapshots o join games g using(game_id) where o.source='espn' and g.season_type='regular' group by 1;
```
| season | n | ML null | spread null | spread price null | total null | O/U price null |
|---|---|---|---|---|---|---|
| 2025-26 | 979 | 0.6 (6) | 0.1 (1) | 0.1 | 0.1 (1) | 0.1 |
| 2026-27 | 188 | 0 | 0 | 0 | 0 | 0 |

All 7 partial rows are on ESPN's first coverage days, 2025-11-24..26 (for example UTA @ GSW 2025-11-24 has only a total, and POR @ MIL has no total). After that, every field is filled. No `0` or |price| < 100 values (`sum(home_ml between -99 and 99)` = 0).

**Ranges** (2025-26 regular): spread −24.5..+21.5, total 207.5..254.5, home ML −9000..+1300. The extreme rows (|ML| ≥ 2000 or |spread| ≥ 20, 23 games) are all 2026-03-14 or later, so they reflect late-season tanking (e.g. UTA @ OKC 2026-04-05, −24.5 / −9000). They're plausible, not parse errors.

**Vig** (via `odds_math.american_to_prob`, script `agentB/calib.py`): ML overround mean 1.0425 (range 1.0275–1.0617), spread 1.0471, total 1.0474. No rows have an overround below 1 or above 1.10. Standard DK.

## 6. Spread-sign check

ESPN `spread` is stored as `home_spread` (negative = home favoured).

```sql
-- ML favourite vs spread favourite (excluding spread = 0, none exist)
select g.season, count(*), sum((home_ml<away_ml)=(home_spread<0)), sum(home_ml<0 and away_ml<0)
from odds_snapshots o join games g using(game_id)
where source='espn' and home_ml is not null and away_ml is not null and home_spread is not null and home_spread<>0 group by 1;
-- cover / over / residuals
select g.season, g.season_type, count(*),
 round(100.0*sum(home_pts-away_pts+home_spread>0)/sum(home_pts-away_pts+home_spread<>0),1) home_cover_pct,
 sum(home_pts-away_pts+home_spread=0) pushes,
 round(100.0*sum(home_pts+away_pts>total)/sum(home_pts+away_pts<>total),1) over_pct,
 round(avg(abs(home_pts-away_pts+home_spread)),2) mae_spread,
 round(avg(abs(home_pts-away_pts-home_spread)),2) mae_if_sign_flipped,
 round(avg(home_pts-away_pts+home_spread),2) mean_resid, round(avg(abs(home_pts+away_pts-total)),2) mae_total
from games g join odds_snapshots o on o.game_id=g.game_id and o.source='espn'
where g.status='final' and home_spread is not null and total is not null group by 1,2;
-- favourite win rate
select g.season, round(100.0*avg((home_spread<0)=(home_pts>away_pts)),1)
from games g join odds_snapshots o on o.game_id=g.game_id and o.source='espn' where home_spread<>0 group by 1;
```

| season | type | n | ML/spread fav agree | home cover % | over % | mean \|margin + spread\| | if sign flipped | mean resid | mean \|total resid\| | spread fav wins |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-26 | regular | 977 | 972/973 (99.9%) | 50.8 | 48.4 | 11.06 | 18.36 | −0.28 | 14.68 | 69.3% |
| 2026-27 | regular | 188 | 188/188 (100%) | 48.9 | 46.3 | 11.26 | 17.96 | +0.41 | 14.03 | 72.3% |

**Verdict: the sign is correct.** Cover and over rates are around 50%, the residual mean is around 0, and flipping the sign would raise the error by 7 points. Pushes: 0, because every line is a half-point.

- **Both MLs negative:** 46 games in 2025-26 and 12 in 2026-27, all on ±1.5 near-pick'em lines. Example: 0022600216 OKC @ POR 2026-11-10, ML −101 home / −117 away, spread +1.5. The ML favourite (OKC, −117) is the spread favourite, so it is consistent, not a bug. 0 games have both MLs positive.
- **The single disagreement:** 0022500872 MEM @ IND 2026-03-01, spread IND −1.5 at **+102** / MEM −122, ML IND −108 / MEM −112. The prices are shaded, so this is effectively a pick'em in which the ML leans MEM and the spread leans IND. It's a real market quirk, not a sign flip. `nba status` reports this as `fav_disagree=1`, which answers C's check: about 0 across the full season.
- **ML calibration** (devigged home prob, regular season 2025-26 + 2026-27, `agentB/calib.py`):

| bucket | n | mean p | actual |
|---|---|---|---|
| ≤0.20 | 83 | 0.141 | 0.096 |
| 0.20–0.35 | 166 | 0.285 | 0.265 |
| 0.35–0.50 | 233 | 0.427 | 0.433 |
| 0.50–0.65 | 254 | 0.578 | 0.571 |
| 0.65–0.80 | 260 | 0.724 | 0.754 |
| >0.80 | 165 | 0.862 | 0.855 |

  Well calibrated, with a mild favourite–longshot bias at the tails. Brier: 0.1948 (2025-26) and 0.1868 (2026-27). That is the bar any model in Phase 2 has to clear.

## 7. Date sanity, neutral sites, data glitches

- **Date ranges** (§2) match the real calendars: regular seasons end 2024-04-14, 2025-04-13 and 2026-04-12. No game is on 2025-11-27 (Thanksgiving) or 2026-11-03 (Election Day), as scheduled. Rest gaps: the maximum is 9 days (All-Star break), there are 0 same-day doubles, and 17–18% of team-games are the second half of a back-to-back.
  ```sql
  with r as (select team_id, season, game_date, julianday(game_date)-julianday(lag(game_date) over
     (partition by team_id, season order by game_date)) gap from team_games where season_type='regular')
  select season, sum(gap=1), round(100.0*sum(gap=1)/count(*),1), max(gap), sum(gap=0) from r group by season;
  ```
- **Games on dates where ESPN had none / ESPN events with no NBA game:** none. In every backfilled regular-season window, all NBA final games got an `espn_event_id` (1230/1230 and 188/188). For 2026-10-20..11-14 I also fetched every scoreboard (script `agentB/datecheck.py`): ESPN shows 188 completed events by ET date, NBA shows 188 final games, 0 dates differ, and 0 ESPN events are unlinked. The UTC→ET conversion works (e.g. on 2025-11-30 ESPN lists three games with `2025-12-01T0xZ` tips under that date, and all three linked).
- **Neutral-site games:** these are the games whose raw `MATCHUP` is `@` on both rows. That's 11 games (script `agentB/neutral.py`, which runs `_ambiguous_games` on the live logs): 2024-25 has 5 (Mexico City WAS–MIA 11-02, Las Vegas Cup semis ×2 on 12-14, Paris SAS–IND ×2), 2025-26 has 5 (Mexico City DAL–DET, Vegas semis ×2, Berlin and London ORL–MEM) and 2026-27 has 1 (DEN–IND 0022600189). All were resolved to a home team, and none was dropped. **2023-24 has 0 flagged**, even though the Vegas semis (0022301229/1230, 2023-12-07) and the Paris game BKN–CLE (0022300527, 2024-01-11) were neutral. The 2023-24 log gives them a normal `vs.`, so they can't be told apart today. There is no neutral-site column in the schema.
- **Data glitch (source):** `0022600231` NOP–DEN 2026-11-14 has DEN `min = 168.0` against NOP's 240.0. The raw `LeagueGameLog` row says `MIN 168`, and it was unchanged after C's re-ingest. It's the only row in the DB with `min < 240` (`select * from team_games where min<240`). Any per-team "per 48" or pace that uses the team's own `min` will inflate DEN's pace by 43% for that game. Use the game's max minutes or regulation + 5·OT.
- **`game_id` order ≠ date order:** Cup knockout games are numbered `…1229/1230`, and the 2026-27 game `0022600015` is dated 2026-10-30. Always sort by `game_date`.

## 8. Anything surprising

1. **ESPN summary 404 aborts the backfill and blocks resume (bug for C).** Details in Next.
2. ESPN has **no 2025-26 lines before 2025-11-24** (250 games), so the 2025-26 against-the-odds backtest covers 979 games, not 1230. The core-API odds endpoint has them.
3. In 2026-27 so far, home teams cover only 48.9% and the over hits 46.3% (n=188). That's within noise (1 SE ≈ 3.6 pts), but worth watching in the dashboard.
4. Home teams per season host 40–42 games, not 41. Neutral games are counted as home games for the designated team.
5. The estimated pace (about 101–102) is about 2 above the NBA's official number, and the gap between the two teams' possession estimates averages 2.3.

## Next

- **Bug (C), high: one ESPN 404 kills the whole `nba espn` run, and resuming can't get past it.**
  - Repro: `uv run nba espn --from 2026-04-13 --to 2026-06-13`. It stops on 2026-04-25 with
    `error: ESPN summary {'event': '401869391'} failed after 3 attempts: 404 Client Error: Not Found for url: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=401869391 (re-run to resume)` (exit 1).
  - Re-running starts at the same event and fails again, because the game is never marked fetched. `--from 2026-04-26` fails the same way on event **401869377**.
  - `curl` shows these summaries really are 404 (NYK @ ATL R1 G4, and a 2026-04-26 game). Their neighbours return 200.
  - Fix in `src/nba_predict/ingest.py:203` / `espn_client._get`: don't retry 4xx. On 404, log a warning, set `espn_event_id` (no snapshot) so resume skips it, and continue. Also commit per date so that work done before an abort is kept (already true).
  - Test: a fake client whose `summary` raises a 404 for one event, and the run still links the others.
- **Feature (data engineering), recommended: ESPN core-API odds fallback, plus open and close lines.**
  - When `pickcenter` is empty, call `…/events/{id}/competitions/{id}/odds` and take the first non-"Live Odds" provider. Store it with its real `book` ("ESPN BET").
  - Store the `open` values as a second snapshot with `is_closing=0`.
  - This fills the 251 missing 2025-26 games (for an 82-game-complete ATS backtest) and gives line movement (open → close) for free. A **"Line movement" view** in the dashboard could then show open vs close spread and total per game, the size of the move, and whether the side the line moved toward covered, with a book column so DK and ESPN BET rows are never mixed silently.
- **Schema (A):** add `games.neutral_site INTEGER` (set from `_ambiguous_games`, plus a small manual list for the 2023-24 Vegas and Paris games above). Exclude those games from home-court stats and the Elo home bonus.
- **Data quality:** compute game minutes as `max(min)` of the two team rows (or 240 + 25·OT) in `analysis/metrics.py`, never a team's own `min`, because of the DEN 168-minute row.
- **Data quality view:** a "Data health" tab in the dashboard showing `nba status`, plus line coverage by month (the table in §5), so gaps like Oct–Nov 2025 are visible next to any backtest result.
