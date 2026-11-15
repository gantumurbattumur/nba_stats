from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from nba_predict.analysis.filters import (
    FilterSpec,
    apply_filter,
    delete_preset,
    list_presets,
    load_preset,
    save_preset,
)
from nba_predict.analysis.metrics import aggregate


def test_last_n_applies_after_other_filters(enriched):
    spec = FilterSpec(teams=["BOS"], venue="away", last_n=3)
    out = apply_filter(enriched, spec)
    assert len(out) == 3 and (out.is_home == 0).all()
    # plan used .nlargest(3, "game_date"); pandas 3 refuses nlargest on str columns
    expected = enriched[(enriched.team_abbr == "BOS") & (enriched.is_home == 0)].sort_values(
        "game_date", ascending=False).head(3)
    assert set(out.game_id) == set(expected.game_id)


def test_last_n_per_team(enriched):
    out = apply_filter(enriched, FilterSpec(last_n=2))
    assert out.groupby("team_abbr").size().to_dict() == {"BOS": 2, "DEN": 2, "LAL": 2, "NYK": 2}
    assert (out.season == "2025-26").all()


# (spec kwargs, predicate that must hold on every returned row, pandas mask for expected count)
CASES = {
    "seasons": (dict(seasons=["2024-25"]), lambda d: d.season == "2024-25"),
    "season_types": (dict(season_types=["regular"]), lambda d: d.season_type == "regular"),
    "date_from": (dict(date_from=date(2025, 10, 25)), lambda d: d.game_date >= "2025-10-25"),
    "date_to": (dict(date_to=date(2024, 10, 25)), lambda d: d.game_date <= "2024-10-25"),
    "as_of": (dict(as_of=date(2024, 10, 25)), lambda d: d.game_date < "2024-10-25"),
    "teams": (dict(teams=["NYK", "DEN"]), lambda d: d.team_abbr.isin(["NYK", "DEN"])),
    "opponents": (dict(opponents=["LAL"]), lambda d: d.opp_abbr == "LAL"),
    "venue_home": (dict(venue="home"), lambda d: d.is_home == 1),
    "venue_away": (dict(venue="away"), lambda d: d.is_home == 0),
    "rest_min": (dict(rest_min=2), lambda d: d.rest_days >= 2),
    "rest_max": (dict(rest_max=1), lambda d: d.rest_days <= 1),
    "b2b_true": (dict(b2b=True), lambda d: d.b2b.astype(bool)),
    "b2b_false": (dict(b2b=False), lambda d: ~d.b2b.astype(bool)),
    "role_fav": (dict(role="favorite"), lambda d: d.is_favorite == 1),
    "role_dog": (dict(role="underdog"), lambda d: d.is_favorite == 0),
    "spread_min": (dict(spread_min=0), lambda d: d.team_spread.notna() & (d.team_spread >= 0)),
    "spread_max": (dict(spread_max=-3), lambda d: d.team_spread.notna() & (d.team_spread <= -3)),
    "total_min": (dict(total_min=225), lambda d: d.total_line.notna() & (d.total_line >= 225)),
    "total_max": (dict(total_max=222), lambda d: d.total_line.notna() & (d.total_line <= 222)),
    "result_win": (dict(result="win"), lambda d: d.win.astype(bool)),
    "result_loss": (dict(result="loss"), lambda d: ~d.win.astype(bool)),
    "ats": (dict(ats="cover"), lambda d: d.ats == "cover"),
    "ou": (dict(ou="under"), lambda d: d.ou == "under"),
    "opp_min_win_pct": (dict(opp_min_win_pct=0.5),
                        lambda d: d.opp_win_pct_pre.notna() & (d.opp_win_pct_pre >= 0.5)),
    "opp_net_rank_max": (dict(opp_net_rank_max=2),
                         lambda d: d.opp_net_rank_pre.notna() & (d.opp_net_rank_pre <= 2)),
    "conferences": (dict(conferences=["West"]), lambda d: d.conference == "West"),
    "opp_conferences": (dict(opp_conferences=["East"]), lambda d: d.opp_conference == "East"),
    "divisions": (dict(divisions=["Pacific"]), lambda d: d.division == "Pacific"),
    "months": (dict(months=[11]), lambda d: d.month == 11),
    "weekdays": (dict(weekdays=[0, 6]), lambda d: d.weekday.isin([0, 6])),
}


def test_cases_cover_every_field():
    covered = {k for kw, _ in CASES.values() for k in kw} | {"last_n"}
    assert covered == set(FilterSpec.model_fields)


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_filter_field(enriched, name):
    kwargs, pred = CASES[name]
    out = apply_filter(enriched, FilterSpec(**kwargs))
    assert len(out) > 0, "fixture should exercise this filter"
    assert pred(out).all()
    assert len(out) == int(pred(enriched).fillna(False).sum())


def test_empty_result(enriched):
    out = apply_filter(enriched, FilterSpec(teams=["BOS"], opponents=["BOS"]))
    assert out.empty and list(out.columns) == list(enriched.columns)
    assert aggregate(out, ["net", "record"]).iloc[0]["gp"] == 0


def test_last_n_validation():
    with pytest.raises(ValidationError):
        FilterSpec(last_n=0)


def test_describe():
    assert FilterSpec().describe() == "all games"
    d = FilterSpec(teams=["BOS"], venue="away", last_n=7, opp_min_win_pct=0.5).describe()
    assert "BOS" in d and "away" in d and "last 7" in d and ".500" in d


def test_preset_roundtrip(conn):
    s = FilterSpec(teams=["NYK"], last_n=7, opp_min_win_pct=0.5)
    save_preset(conn, "nyk-l7", s)
    assert load_preset(conn, "nyk-l7") == s and "nyk-l7" in list_presets(conn)
    s2 = FilterSpec(teams=["NYK"], last_n=10, date_from=date(2025, 1, 1))
    save_preset(conn, "nyk-l7", s2)  # overwrite
    assert load_preset(conn, "nyk-l7") == s2
    delete_preset(conn, "nyk-l7")
    assert list_presets(conn) == []
    with pytest.raises(KeyError):
        load_preset(conn, "nyk-l7")


def test_explore_cli(fixture_db, enriched, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from nba_predict import cli
    from nba_predict.config import get_settings

    db = fixture_db.execute("pragma database_list").fetchone()[2]
    monkeypatch.setenv("NBA_DB_PATH", db)
    get_settings.cache_clear()
    try:
        out = tmp_path / "x.csv"
        r = CliRunner().invoke(cli.app, ["explore", "--team", "BOS", "--venue", "away", "--last-n", "3",
                                         "--metrics", "net,record", "--csv", str(out)])
        assert r.exit_code == 0, r.output
        df = pd.read_csv(out)
        assert df.gp[0] == 3 and "net" in df and "record" in df
        save_preset(fixture_db, "bos-away", FilterSpec(teams=["BOS"], venue="away"))
        r = CliRunner().invoke(cli.app, ["explore", "--preset", "bos-away", "--metrics", "win_pct",
                                         "--group-by", "opp_abbr"])
        assert r.exit_code == 0 and "LAL" in r.output and "BOS" not in r.output.split("\n", 1)[1]
    finally:
        get_settings.cache_clear()
