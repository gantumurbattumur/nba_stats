# Phase 1 — Agent C review (Tasks 1–4)

Reviewed `git diff 3636136..6f25be4 -- src tests` against the plan (Global Constraints, Interfaces of Tasks 1–4, Review Focus) and the spec. Branch `dev`.

- Start: `33 passed in 0.19s`
- End: **`75 passed in 0.22s`** (`uv run pytest -q`)

Commits (C):

| commit | message |
|---|---|
| `b83c6f6` | fix: validate identifiers in upsert and make it all-or-nothing |
| `347ba29` | fix: reject invalid American odds; flag ESPN home/away mismatches |
| `1df19f3` | test: phase-1 edge cases (slate/status, live-board fallback, teams, settings, CLI) |
| `d0882f8` | fix: slate skips preseason and All-Star games |

## Findings

| # | Severity | File:line | Issue | Status |
|---|---|---|---|---|
| 1 | High | `src/nba_predict/db.py:152` (`upsert`) | Table/column/key names went into SQL unchecked inside `"..."`. A key like `game_id") do nothing; --` **ran**: the trailing comment swallowed the rest of the statement. Task 11 custom features and the dashboard will be close to this code. | **Fixed.** Identifiers must match `^[A-Za-z_][A-Za-z0-9_]*$` and are all checked before any write, else `ValueError`. |
| 2 | Medium | `src/nba_predict/db.py:185` | With rows of mixed column sets, an error in a later group left the earlier groups in the open transaction, and the caller's next `commit()` saved them. | **Fixed.** Rollback on error, so each `upsert` call is all-or-nothing. |
| 3 | Medium | `src/nba_predict/odds_math.py:10` | `american_to_prob(0)` returned 1.0 and `american_to_decimal(0)` returned 1.0. Any \|odds\| < 100 gave a silent wrong result, which would have fed edges, no-vig odds and `unit_return`. | **Fixed.** Prices strictly between −100 and +100 now raise `ValueError`. |
| 4 | Medium | `src/nba_predict/data/espn_client.py:31` | `parse_pickcenter` would have stored ESPN `0` or out-of-range prices as real odds. None seen in the 188 live rows so far, but the parser is meant to tolerate bad fields. | **Fixed.** `_price()` maps them to `None`. |
| 5 | Medium | `src/nba_predict/ingest.py:175` | When ESPN's home/away is the reverse of the NBA's (a risk at neutral sites), the event was dropped with no message. | **Fixed (warning only).** It logs `home/away reversed vs NBA game …` and is still not linked, so the spread sign is never inverted. The Mexico City game 0022600189 (IND home) matches ESPN today, checked live. |
| 6 | Medium | `src/nba_predict/ingest.py:105,125` | `ingest_slate` stored preseason (`001…`) and All-Star (`003…`) games as `scheduled`. The regular-season log never finalizes them, so they stay scheduled forever. They would also break the Task 7 check "feature rows = `count(*) from games`". | **Fixed.** Both are skipped. `ingest_slate` now returns the number of games upserted. |
| 7 | Medium | `src/nba_predict/data/nba_client.py:60` | `LeagueGameLog` uses its default season type (Regular Season). Playoff (`004`), play-in (`005`) and NBA Cup final (`006`) games from the slate stay `scheduled` and never get box scores or grades. | **Open** (scope decision, see Next). |
| 8 | Low | `src/nba_predict/config.py:11,14` | `.env` and the relative `db_path` both resolve against the current directory. Running `nba` from another folder quietly creates a new empty `data/nba.db` and finds no key. | **Open** (see Next). |
| 9 | Low | `src/nba_predict/ingest.py:29` | `_ambiguous_games` also flags games with a single `@` row (in progress, not yet in the log). That costs one wasted slate lookup per such date. Harmless. | Not fixed (cosmetic). |
| 10 | Low | `src/nba_predict/ingest.py:187` | `pulled_at = <date>T23:59:59Z` is 6:59pm ET, earlier than the real close for late tips. It follows the plan, and the closing line is chosen by `is_closing`, not the time. | Not fixed (as specified). Task 13 CLV must not order ESPN against Odds API by `pulled_at`. |
| 11 | Info | `tests/test_espn.py` | The plan's `test_ingest_matches_and_is_resumable` assertion `summary_calls == n1` assumed every fetched game has a line. A's version (2 calls, 1 snapshot, the no-line game still linked) is the correct reading. | OK |

Verified with no changes needed:
- **Interfaces** match the plan for Tasks 1–4: `get_settings`, `connect`, `init_schema`, `upsert`, `read_df`, `Team`/`TEAMS`/`by_*`, the odds_math functions, `NbaClient(delay_s, retries=3)` with `.season_team_logs` and `.slate` and their dict keys, `IngestError`, `parse_team_logs`, `ingest_season`, `ingest_slate`, `EspnClient(delay_s)` with `.events` and `.summary`, `parse_pickcenter` keys, and `ingest_espn_lines(conn, client, date_from, date_to, only_missing=True)`. The CLI provides `init-db`, `ingest --season` (repeatable), `slate --date` and `espn --from/--to` (plus `--refetch`).
- **Schema** covers every table and column in the plan's schema table.
- **Analysis-only constraint:** grepping `src/`, `tests/`, `pyproject.toml`, `README.md` and `.env.example` for bankroll, stake, kelly, wager, bank, wallet or sportsbook finds only the disclaimers. A test now makes sure `Settings` never gains such a field.
- **Teams:** the 30 teams equal `nba_api.stats.static.teams` (id, abbreviation, full name). The ESPN abbreviations NY, GS, SA, NO, UTAH, WSH, BKN and PHX are confirmed against the live ESPN scoreboard. Divisions are 6 × 5, every lookup round-trips, and Odds API names are the NBA full names.
- **Spread sign** is home-perspective, checked on real data in B's DB (read-only). Across 188 ESPN snapshots, the spread sign agrees with the ML favorite every time (0 mismatches). The spread favorite won 72.3% of games and the home side covered 48.9%. An inverted sign would push both numbers to extremes.
- **Live smoke** on a temp DB (`NBA_DB_PATH=<scratchpad>`): `nba slate` worked for 2026-11-14 (ScoreboardV2), today (live board) and 2026-11-16. ScoreboardV2's GameHeader still works with nba_api 1.12.0.
- **Re-run of B-style checks** on `data/nba.db` (read-only):
  - final games per full season: 1230/1230/1230, with team_games 2460 each
  - 0 games without exactly 2 team rows
  - 0 home-flag or points mismatches between `games` and `team_games`
  - 0 cases of one team playing twice on a date
  - home win rate: .543 / .544 / .554 / .527 (2026-27 so far)
- **Review Focus #2:** `test_summary_without_pickcenter` exists and passes on the real 2024-25 summary 401705127, and also on `{}`.

## Tests added (42 new test cases; 33 → 75)

- `tests/test_db.py`:
  - partial row keeps other columns, including `espn_event_id` after a re-ingest
  - mixed column sets in one call
  - unsafe table/column/key names rejected (3 cases)
  - all-or-nothing across column groups
- `tests/test_odds_math.py`:
  - ±100 is even money
  - invalid odds 0, ±50 and ±99 raise
  - `best_price` with mixed signs
- `tests/test_espn.py`:
  - home-underdog spread is positive
  - pick'em `0` spread is kept, not treated as missing
  - invalid or blank prices become `None`
  - ET date boundaries (6 cases across EDT, EST and the end of DST, including a 10:30pm tip)
  - ESPN 503 → 3 attempts → `IngestError`
  - home/away reversal warns and doesn't link
- `tests/test_ingest_nba.py`:
  - re-ingest and slate keep `espn_event_id`
  - scoreboard-final game stays `scheduled`, then `ingest_season` promotes it
  - postponed → rescheduled date and season
  - in-progress (WL missing) game skipped
  - status mapping, including PPD
  - live board used for today
  - stale live board falls back to V2
  - preseason and All-Star skipped
- `tests/test_teams.py`:
  - equals the nba_api static teams
  - every lookup round-trips, ignoring case and whitespace, with `None`/empty handled
  - 6 divisions × 5 teams
- `tests/test_config.py` (new):
  - defaults
  - unprefixed `ODDS_API_KEY` plus `NBA_` fields, with JSON windows
  - `.env` read from the current directory
  - no staking fields
- `tests/test_cli.py` (new), all on a temp DB:
  - `init-db` uses `NBA_DB_PATH`
  - `slate` output, and `IngestError` → exit 1 with no traceback
  - `espn` over a range with no games makes zero HTTP calls

Final: `75 passed in 0.22s`.

## Verdict on A's 8 deviations

1. **`test_devig` → `approx(0.57983)`: correct.** Implied odds −150 → 0.6 and +130 → 100/230 = 0.434783, so 0.6 / 1.034783 = 0.579832. The plan's `a > 0.58` was simply wrong.
2. **Neutral-site home lookup: correct.** The order is: existing `games` row, then `client.slate(date)`, then skip with a warning. Live game 0022600189 (DEN vs IND, Mexico City) resolved to IND as home, the same as ESPN. Tests cover both paths. I added the case where ESPN's home/away is reversed (#5).
3. **In-progress games (WL NaN) skipped: correct.** Box scores are partial until the game ends. The next ingest picks the game up. Test added.
4. **`ingest_slate` never touches final rows, and a scoreboard-final game not yet in the log is stored `scheduled`: correct.** `final` should mean "scores and box score present", and grading (Task 13) needs `home_pts`. `nba daily` runs ingest before slate, so the gap only lasts until the next ingest. Tests cover promotion to final and that `espn_event_id` survives. The only real hole is #7 (playoff and Cup-final games).
5. **Retries: 3 attempts total (backoff 1s, 2s), ValueError retried: acceptable.** This matches the plan test ("IngestError after 3 attempts"); a 4s sleep would only happen with a 4th attempt. Retrying `ValueError` is right, because stats.nba.com returns HTML or blank responses when it blocks you, and that raises a `JSONDecodeError`, which is a `ValueError`. A `KeyError` from a reshaped live board is not retried or wrapped, which is minor.
6. **`ingest_espn_lines` skip rules: correct, and better than the plan.**
   - Also skipping games that already have an `espn_event_id` stops every run from re-fetching the line-less 2023-25 seasons.
   - Fetching only completed events means the line is the closing line.
   - The ET conversion is verified at the date boundaries.
   - Zero HTTP calls for empty dates is verified, including through the CLI.
   - `--refetch` gives a way out if ESPN fills in a line later.
7. **`upsert` groups rows by column set: correct and necessary.** Without it, re-running `ingest_season` would set `espn_event_id` to NULL. Tests are pinned. I made it all-or-nothing and identifier-safe (#1, #2).
8. **Settings, `ODDS_API_KEY` alias, `.env` read from the current directory: correct as written.** `NBA_ODDS_API_KEY` also works. Reading from the current directory is a usability trap (#8).

## Next

- **Bug fix (recommended for Phase 2, owner A):** add post-season games to ingest (#7). In `NbaClient.season_team_logs(season, season_type="Regular Season")`, loop `ingest_season` over `("Regular Season", "PlayIn", "Playoffs")` when `season == current_season` or when backfilling. Also look up the NBA Cup final (`006…`) through the `IST` season type, or mark it `excluded`. Until then, `ingest_slate` should store `004/005/006` games with status `excluded` (or skip them), so `predict`/`grade` don't carry ungradable rows. A test should check that an `004…` slate game ends up `final` after ingest.
- **Feature (low effort):** make `db_path` and `.env` resolve from the project root (`Path(__file__).parents[2]`) when relative, or set `NBA_HOME`. Print the resolved DB path in every CLI command, so running from another folder can't silently create an empty DB (#8).
- **Feature:** `parse_pickcenter` should prefer `pointSpread.home.close.line/odds` and `moneyline.home.close.odds` when present. These are unambiguous closing fields, and they also hold the **open** line (`…open…`). Store the open line as an extra ESPN snapshot (`is_closing=0`, `pulled_at=<date>T12:00:00Z`) to get line movement and CLV-style evaluation on 2025-26 for free.
- **Check for B:** after the full 2025-26 ESPN backfill, re-run `select count(*) from odds_snapshots where source='espn' and ((home_spread<0) != (home_ml<away_ml))`. It should be about 0, which confirms the sign across the whole season. Also grep the ingest log for `home/away reversed`.
