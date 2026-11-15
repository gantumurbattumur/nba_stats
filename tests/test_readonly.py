import pytest

from nba_predict.db import QueryError, readonly_query


def _path(conn):
    return conn.execute("pragma database_list").fetchone()[2]


def _games(conn):
    return conn.execute("select count(*) from games").fetchone()[0]


def test_select_ok(fixture_db):
    df = readonly_query(_path(fixture_db), "select game_id, home_pts from games where season = ? order by 1",
                        ("2024-25",))
    assert len(df) == 24 and list(df.columns) == ["game_id", "home_pts"]
    cte = readonly_query(_path(fixture_db), "with recursive n(x) as (select 1 union all select x+1 from n where x<5) "
                         "select sum(x) s from n")
    assert cte.s[0] == 15
    assert readonly_query(_path(fixture_db), "select count(*) c from games;").c[0] == 50  # trailing ; ok


def test_table_info_allowed(fixture_db):
    df = readonly_query(_path(fixture_db), "select name from pragma_table_info('games')")
    assert "game_id" in set(df.name)


def test_max_rows(fixture_db):
    df = readonly_query(_path(fixture_db), "select * from games", max_rows=7)
    assert len(df) == 7 and df.attrs["truncated"] is True


@pytest.mark.parametrize("sql", [
    "DELETE FROM games",
    "DROP TABLE games",
    "INSERT INTO games (game_id) VALUES ('x')",
    "UPDATE games SET home_pts = 0",
    "ATTACH DATABASE ':memory:' AS other",
    "PRAGMA writable_schema=1",
    "select 1; delete from games",
    "CREATE TABLE z (a)",
    "select load_extension('x')",
])
def test_rejects_writes(fixture_db, sql):
    before = _games(fixture_db)
    with pytest.raises(QueryError):
        readonly_query(_path(fixture_db), sql)
    assert _games(fixture_db) == before


def test_bad_sql_is_query_error(fixture_db):
    with pytest.raises(QueryError):
        readonly_query(_path(fixture_db), "select nope from games")
