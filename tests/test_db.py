import pytest

from nba_predict.db import init_schema, upsert


def test_schema_idempotent(conn):
    init_schema(conn)
    init_schema(conn)
    names = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert {"games", "team_games", "odds_snapshots", "api_usage", "predictions", "value_flags",
            "filter_presets", "custom_features", "backtest_results"} <= names


def test_upsert_updates_not_duplicates(conn):
    row = dict(game_id="g1", season="2026-27", game_date="2026-11-01", home_team_id=1,
               away_team_id=2, status="scheduled")
    upsert(conn, "games", [row], ["game_id"])
    upsert(conn, "games", [{**row, "status": "final", "home_pts": 110, "away_pts": 100}], ["game_id"])
    rows = conn.execute("select status, home_pts from games").fetchall()
    assert [tuple(r) for r in rows] == [("final", 110)]


def test_upsert_empty_rows(conn):
    assert upsert(conn, "games", [], ["game_id"]) == 0


def test_read_df(conn):
    from nba_predict.db import read_df
    upsert(conn, "games", [dict(game_id="g1", season="2026-27", game_date="2026-11-01",
                                home_team_id=1, away_team_id=2, status="scheduled")], ["game_id"])
    df = read_df(conn, "select game_id from games where season = ?", ("2026-27",))
    assert list(df.game_id) == ["g1"]


def test_upsert_partial_row_keeps_other_columns(conn):
    full = dict(game_id="g1", season="2026-27", game_date="2026-11-01", home_team_id=1,
                away_team_id=2, status="final", home_pts=110, away_pts=100)
    upsert(conn, "games", [full], ["game_id"])
    upsert(conn, "games", [dict(game_id="g1", espn_event_id="401")], ["game_id"])
    upsert(conn, "games", [full], ["game_id"])  # re-ingest without espn_event_id
    r = conn.execute("select status, home_pts, espn_event_id from games").fetchone()
    assert tuple(r) == ("final", 110, "401")


def test_upsert_mixed_column_sets_in_one_call(conn):
    a = dict(game_id="g1", season="2026-27", status="final", home_pts=110)
    b = dict(game_id="g2", season="2026-27", status="scheduled")
    upsert(conn, "games", [a, b], ["game_id"])
    upsert(conn, "games", [dict(game_id="g1", status="final")], ["game_id"])
    rows = dict(conn.execute("select game_id, home_pts from games").fetchall())
    assert rows == {"g1": 110, "g2": None}


@pytest.mark.parametrize("table,cols,key", [
    ('games" (game_id) values (1); drop table games; --', ["game_id"], ["game_id"]),
    ("games", ['status" = 1; --'], ["game_id"]),
    ("games", ["game_id"], ['game_id") do nothing; --']),
])
def test_upsert_rejects_unsafe_identifiers(conn, table, cols, key):
    row = {c: "x" for c in cols} | {"game_id": "g1"}
    with pytest.raises(ValueError):
        upsert(conn, table, [row], key)
    assert conn.execute("select count(*) from games").fetchone()[0] == 0


def test_upsert_is_atomic_across_column_groups(conn):
    import sqlite3
    good = dict(game_id="g1", season="2026-27", status="final")
    bad = dict(game_id="g2", no_such_column=1)
    with pytest.raises(sqlite3.OperationalError):
        upsert(conn, "games", [good, bad], ["game_id"])
    conn.commit()  # a later commit by any caller must not persist the half-written call
    assert conn.execute("select count(*) from games").fetchone()[0] == 0


def test_migration_adds_season_type(tmp_path):
    """A DB created before season_type existed gets the column (default 'regular'),
    and post-season ids are labelled from their prefix."""
    import sqlite3

    from nba_predict.db import connect

    p = tmp_path / "old.db"
    old = sqlite3.connect(p)
    old.executescript("""
        CREATE TABLE games (game_id TEXT PRIMARY KEY, season TEXT, game_date TEXT,
            home_team_id INTEGER, away_team_id INTEGER, home_pts INTEGER, away_pts INTEGER,
            status TEXT, espn_event_id TEXT);
        CREATE TABLE team_games (game_id TEXT, team_id INTEGER, opp_team_id INTEGER,
            season TEXT, game_date TEXT, is_home INTEGER, wl TEXT, PRIMARY KEY (game_id, team_id));
        INSERT INTO games (game_id, status) VALUES ('0022500001', 'final'), ('0042500101', 'final');
        INSERT INTO team_games (game_id, team_id) VALUES ('0022500001', 1), ('0052500101', 2);
    """)
    old.commit()
    old.close()
    c = connect(p)
    init_schema(c)
    init_schema(c)  # idempotent
    assert dict(c.execute("select game_id, season_type from games").fetchall()) == {
        "0022500001": "regular", "0042500101": "playoffs"}
    assert dict(c.execute("select game_id, season_type from team_games").fetchall()) == {
        "0022500001": "regular", "0052500101": "playin"}
