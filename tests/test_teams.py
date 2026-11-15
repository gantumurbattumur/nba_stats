from nba_predict.teams import TEAMS, by_abbr, by_espn_abbr, by_nba_id, by_odds_name


def test_thirty_teams_unique():
    assert len(TEAMS) == 30 and len({t.nba_id for t in TEAMS}) == 30
    assert sum(t.conference == "East" for t in TEAMS) == 15
    assert len({t.espn_abbr for t in TEAMS}) == 30 and len({t.abbr for t in TEAMS}) == 30


def test_lookups():
    assert by_abbr("BOS").nba_id == 1610612738
    assert by_odds_name("Los Angeles Clippers").abbr == "LAC"
    assert by_espn_abbr("NY").abbr == "NYK"
    assert by_espn_abbr("GS").abbr == "GSW" and by_espn_abbr("UTAH").abbr == "UTA"
    assert by_odds_name("Nowhere Team") is None
    assert by_nba_id(1610612747).abbr == "LAL"
    assert by_espn_abbr("WSH").abbr == "WAS" and by_espn_abbr("SA").abbr == "SAS"
    assert by_espn_abbr("NO").abbr == "NOP"


def test_matches_nba_api_static_teams():
    from nba_api.stats.static import teams as static

    ref = {t["id"]: (t["abbreviation"], t["full_name"]) for t in static.get_teams()}
    assert {t.nba_id: (t.abbr, t.name) for t in TEAMS} == ref


def test_every_team_reachable_by_each_lookup():
    for t in TEAMS:
        assert by_nba_id(t.nba_id) is t and by_abbr(t.abbr) is t
        assert by_espn_abbr(t.espn_abbr) is t and by_odds_name(t.odds_name) is t
    assert by_odds_name("  los angeles lakers ").abbr == "LAL"   # tolerant of case/space
    assert by_espn_abbr("gs").abbr == "GSW"
    assert by_abbr("") is None and by_espn_abbr(None) is None and by_nba_id(None) is None


def test_divisions_have_five_teams():
    from collections import Counter
    c = Counter((t.conference, t.division) for t in TEAMS)
    assert len(c) == 6 and set(c.values()) == {5}
