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


@pytest.mark.parametrize("sql", [
    "WITH x AS (SELECT 1) DELETE FROM games",
    "WITH x AS (SELECT 1) INSERT INTO games (game_id) SELECT 'z' FROM x",
    "VACUUM",
    "CREATE TEMP TABLE t (x)",
    "CREATE TEMP VIEW v AS SELECT 1",
    "SAVEPOINT a",
    "BEGIN IMMEDIATE",
    "ANALYZE",
    "PRAGMA journal_mode = delete",
    "SELECT * FROM pragma_database_list",
    "SELECT * FROM pragma_user_version",
    "SELECT * FROM pragma_integrity_check",
])
def test_rejects_other_side_effects(fixture_db, sql):
    with pytest.raises(QueryError):
        readonly_query(_path(fixture_db), sql)


def test_vacuum_into_writes_no_file(fixture_db, tmp_path):
    out = tmp_path / "copy.db"
    with pytest.raises(QueryError):
        readonly_query(_path(fixture_db), f"VACUUM INTO '{out}'")
    assert not out.exists()


@pytest.mark.parametrize("sql", [
    "select length(randomblob(900000000))",
    "select zeroblob(500000000)",
    "select length(replace(hex(zeroblob(4000000)), '0', '00000000'))",
])
def test_huge_values_rejected(fixture_db, sql):
    # each of these allocated ~0.1-1 GB in-process (dashboard memory DoS)
    with pytest.raises(QueryError):
        readonly_query(_path(fixture_db), sql)


def test_oversized_printf_is_null_not_allocated(fixture_db):
    df = readonly_query(_path(fixture_db), "select length(printf('%.*c', 900000000, 'x')) n")
    assert df.n.isna().all()


def test_result_size_budget(fixture_db):
    # 5 MB per row × 200 rows = 1 GB of results: must stop well before that
    sql = ("with recursive r(i) as (select 1 union all select i + 1 from r where i < 200) "
           "select i, randomblob(5000000) from r")
    with pytest.raises(QueryError, match="too large"):
        readonly_query(_path(fixture_db), sql, max_bytes=50_000_000)


def test_runaway_query_times_out(fixture_db):
    sql = "with recursive r(i) as (select 1 union all select i + 1 from r) select count(*) from r"
    with pytest.raises(QueryError):
        readonly_query(_path(fixture_db), sql, timeout_s=0.5)
