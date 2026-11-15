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
