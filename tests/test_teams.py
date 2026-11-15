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
