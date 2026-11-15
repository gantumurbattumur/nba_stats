import pytest

from nba_predict.analysis.custom import (
    add_feature,
    delete_feature,
    evaluate,
    list_features,
    set_use_in_model,
    with_custom,
)
from nba_predict.db import QueryError
from nba_predict.features.build import build_feature_table


@pytest.fixture
def feats(fixture_db, enriched):
    return build_feature_table(fixture_db)


def _path(conn):
    return conn.execute("pragma database_list").fetchone()[2]


def test_custom_feature_roundtrip(fixture_db, feats):
    add_feature(fixture_db, "l5_split_gap", "h_split_l5_net - a_split_l10_net")
    lf = list_features(fixture_db)
    assert list(lf.name) == ["l5_split_gap"] and lf.use_in_model[0] == 0
    ev = evaluate(_path(fixture_db)).set_index("game_id")
    exp = (feats.h_split_l5_net - feats.a_split_l10_net).set_axis(feats.game_id)
    ok = exp.notna()
    assert (ev.loc[exp.index[ok], "l5_split_gap"] - exp[ok]).abs().max() < 1e-9
    assert ev.loc[exp.index[~ok], "l5_split_gap"].isna().all()


def test_use_in_model_and_with_custom(fixture_db, feats):
    add_feature(fixture_db, "elo_edge", "elo_prob - 0.5", use_in_model=True)
    add_feature(fixture_db, "rest_gap", "h_rest - a_rest")
    out = with_custom(feats, _path(fixture_db))
    assert {"elo_edge", "rest_gap"} <= set(out.columns) and len(out) == len(feats)
    only = with_custom(feats, _path(fixture_db), only_model=True)
    assert "elo_edge" in only and "rest_gap" not in only
    set_use_in_model(fixture_db, "elo_edge", False)
    assert "elo_edge" not in with_custom(feats, _path(fixture_db), only_model=True)
    delete_feature(fixture_db, "rest_gap")
    assert list(list_features(fixture_db).name) == ["elo_edge"]
    assert evaluate(_path(fixture_db), ["elo_edge"]).columns.tolist() == ["game_id", "elo_edge"]


@pytest.mark.parametrize("expr", ["nonexistent_col + 1", "1); delete from games; --",
                                  "(select count(*) from games)) ; drop table games; select (1"])
def test_bad_expression_rejected(fixture_db, feats, expr):
    n = fixture_db.execute("select count(*) from games").fetchone()[0]
    with pytest.raises(QueryError):
        add_feature(fixture_db, "bad", expr)
    assert list_features(fixture_db).empty
    assert fixture_db.execute("select count(*) from games").fetchone()[0] == n


@pytest.mark.parametrize("name", ["Bad", "1abc", "has space", "x" * 42, "h_all_l5_net", ""])
def test_bad_names_rejected(fixture_db, feats, name):
    with pytest.raises(ValueError):
        add_feature(fixture_db, name, "1")
    assert list_features(fixture_db).empty


def test_no_custom_features_is_noop(fixture_db, feats):
    assert with_custom(feats, _path(fixture_db)).shape == feats.shape


@pytest.mark.parametrize("expr", [
    # comment swallows the rest of the query: validates as 2 columns, then breaks evaluate()
    "h_rest) AS v FROM game_features --",
    "h_rest) AS v FROM game_features /*",
    # leaves the parentheses: reads another table / changes the row set
    "1) AS v FROM games UNION SELECT game_id, (1",
    "h_rest) AS v FROM game_features WHERE (0",
    "h_rest), (a_rest",
    "h_rest; select 1",
])
def test_expression_cannot_escape_its_parentheses(fixture_db, feats, expr):
    with pytest.raises(QueryError):
        add_feature(fixture_db, "esc", expr)
    assert list_features(fixture_db).empty


@pytest.mark.parametrize("expr", [
    "case when h_rest > a_rest then 1 else 0 end",
    "coalesce(h_all_l5_net, 0) - (a_all_l5_net)",
    "'a)b' || ')' || \"h_rest\"",      # parens inside literals / quoted identifiers are fine
    "(select avg(h_rest) from game_features)",
])
def test_legit_expressions_accepted(fixture_db, feats, expr):
    add_feature(fixture_db, "ok_feat", expr)
    assert "ok_feat" in evaluate(_path(fixture_db)).columns


def test_one_broken_feature_does_not_break_the_rest(fixture_db, feats, caplog):
    # a stored feature can go stale (e.g. features rebuilt with windows=(3,) drop the l5 columns)
    add_feature(fixture_db, "good", "h_rest - a_rest")
    add_feature(fixture_db, "uses_l5", "h_all_l5_net")
    build_feature_table(fixture_db, windows=(3,))
    with caplog.at_level("WARNING"):
        ev = evaluate(_path(fixture_db))
    assert "good" in ev.columns and "uses_l5" not in ev.columns
    assert any("uses_l5" in r.getMessage() for r in caplog.records)
