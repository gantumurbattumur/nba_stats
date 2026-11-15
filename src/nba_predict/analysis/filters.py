"""One filter engine for dashboard, CLI and models: ``FilterSpec`` → pandas mask, plus presets.

Row filters apply first, then ``last_n`` per team. Odds-based filters drop rows with NULL odds;
opponent-strength filters drop rows with NULL ``opp_*_pre``.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field


class FilterSpec(BaseModel):
    seasons: list[str] | None = None
    season_types: list[Literal["regular", "playin", "playoffs"]] | None = None
    date_from: dt.date | None = None
    date_to: dt.date | None = None
    as_of: dt.date | None = None  # strictly before
    teams: list[str] | None = None
    opponents: list[str] | None = None
    venue: Literal["all", "home", "away"] = "all"
    last_n: int | None = Field(default=None, ge=1)
    rest_min: int | None = None
    rest_max: int | None = None
    b2b: bool | None = None
    role: Literal["any", "favorite", "underdog"] = "any"
    spread_min: float | None = None
    spread_max: float | None = None
    total_min: float | None = None
    total_max: float | None = None
    result: Literal["any", "win", "loss"] = "any"
    ats: Literal["any", "cover", "no_cover", "push"] = "any"
    ou: Literal["any", "over", "under", "push"] = "any"
    opp_min_win_pct: float | None = None
    opp_net_rank_max: int | None = None
    conferences: list[str] | None = None
    opp_conferences: list[str] | None = None
    divisions: list[str] | None = None
    months: list[int] | None = None
    weekdays: list[int] | None = None  # 0 = Monday

    def describe(self) -> str:
        p: list[str] = []
        if self.teams:
            p.append(",".join(self.teams))
        if self.opponents:
            p.append("vs " + ",".join(self.opponents))
        if self.venue != "all":
            p.append(self.venue)
        if self.last_n:
            p.append(f"last {self.last_n}")
        if self.seasons:
            p.append("seasons " + ",".join(self.seasons))
        if self.season_types:
            p.append(",".join(self.season_types))
        if self.date_from or self.date_to:
            p.append(f"{self.date_from or '…'} to {self.date_to or '…'}")
        if self.as_of:
            p.append(f"before {self.as_of}")
        if self.rest_min is not None or self.rest_max is not None:
            p.append(f"rest {self.rest_min if self.rest_min is not None else 0}"
                     f"–{self.rest_max if self.rest_max is not None else '7'}")
        if self.b2b is not None:
            p.append("b2b" if self.b2b else "not b2b")
        if self.role != "any":
            p.append(self.role)
        if self.spread_min is not None or self.spread_max is not None:
            p.append(f"spread {self.spread_min if self.spread_min is not None else '…'}"
                     f" to {self.spread_max if self.spread_max is not None else '…'}")
        if self.total_min is not None or self.total_max is not None:
            p.append(f"total {self.total_min if self.total_min is not None else '…'}"
                     f" to {self.total_max if self.total_max is not None else '…'}")
        for name in ("result", "ats", "ou"):
            v = getattr(self, name)
            if v != "any":
                p.append(f"{name}={v}")
        if self.opp_min_win_pct is not None:
            p.append(f"opp win% ≥ {self.opp_min_win_pct:.3f}".replace("0.", "."))
        if self.opp_net_rank_max is not None:
            p.append(f"opp net rank ≤ {self.opp_net_rank_max}")
        for name in ("conferences", "opp_conferences", "divisions", "months", "weekdays"):
            v = getattr(self, name)
            if v:
                p.append(f"{name}={','.join(map(str, v))}")
        return " · ".join(p) if p else "all games"


def apply_filter(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)

    def isin(col, vals):
        return df[col].isin(vals)

    if spec.seasons:
        m &= isin("season", spec.seasons)
    if spec.season_types:
        m &= isin("season_type", spec.season_types)
    if spec.date_from:
        m &= df.game_date >= spec.date_from.isoformat()
    if spec.date_to:
        m &= df.game_date <= spec.date_to.isoformat()
    if spec.as_of:
        m &= df.game_date < spec.as_of.isoformat()
    if spec.teams:
        m &= isin("team_abbr", [t.upper() for t in spec.teams])
    if spec.opponents:
        m &= isin("opp_abbr", [t.upper() for t in spec.opponents])
    if spec.venue != "all":
        m &= df.is_home == (1 if spec.venue == "home" else 0)
    if spec.rest_min is not None:
        m &= df.rest_days >= spec.rest_min
    if spec.rest_max is not None:
        m &= df.rest_days <= spec.rest_max
    if spec.b2b is not None:
        m &= df.b2b.astype(bool) == spec.b2b
    if spec.role != "any":
        m &= (df.is_favorite == (1 if spec.role == "favorite" else 0)).fillna(False).astype(bool)
    for col, lo, hi in (("team_spread", spec.spread_min, spec.spread_max),
                        ("total_line", spec.total_min, spec.total_max)):
        if lo is not None:
            m &= (df[col] >= lo).fillna(False)  # NaN odds never pass
        if hi is not None:
            m &= (df[col] <= hi).fillna(False)
    if spec.result != "any":
        m &= df.win.astype(bool) == (spec.result == "win")
    if spec.ats != "any":
        m &= df.ats == spec.ats
    if spec.ou != "any":
        m &= df.ou == spec.ou
    if spec.opp_min_win_pct is not None:
        m &= (df.opp_win_pct_pre >= spec.opp_min_win_pct).fillna(False)
    if spec.opp_net_rank_max is not None:
        m &= (df.opp_net_rank_pre <= spec.opp_net_rank_max).fillna(False)
    if spec.conferences:
        m &= isin("conference", spec.conferences)
    if spec.opp_conferences:
        m &= isin("opp_conference", spec.opp_conferences)
    if spec.divisions:
        m &= isin("division", spec.divisions)
    if spec.months:
        m &= isin("month", spec.months)
    if spec.weekdays:
        m &= isin("weekday", spec.weekdays)
    out = df[m.astype(bool)]
    if spec.last_n:
        keep = (out.sort_values("game_date", ascending=False, kind="stable")
                .groupby("team_abbr", sort=False).head(spec.last_n).index)
        out = out.loc[out.index.isin(keep)]
    return out


def save_preset(conn: sqlite3.Connection, name: str, spec: FilterSpec) -> None:
    conn.execute(
        "insert into filter_presets (name, spec_json, created_at) values (?, ?, ?) "
        "on conflict (name) do update set spec_json = excluded.spec_json, created_at = excluded.created_at",
        (name, spec.model_dump_json(), dt.datetime.now(dt.UTC).isoformat(timespec="seconds")))
    conn.commit()


def load_preset(conn: sqlite3.Connection, name: str) -> FilterSpec:
    row = conn.execute("select spec_json from filter_presets where name = ?", (name,)).fetchone()
    if row is None:
        raise KeyError(f"no preset named {name!r}")
    return FilterSpec.model_validate_json(row[0])


def list_presets(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("select name from filter_presets order by name")]


def delete_preset(conn: sqlite3.Connection, name: str) -> None:
    conn.execute("delete from filter_presets where name = ?", (name,))
    conn.commit()
