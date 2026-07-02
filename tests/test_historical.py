"""Tests for the martj42 history parser (stdlib csv, no pandas)."""

from __future__ import annotations

from datetime import date

from worldcup.data.historical import HistoricalMatch, parse_results_csv

CSV = (
    "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
    "2026-06-11,Mexico,South Africa,2,0,FIFA World Cup,Mexico City,Mexico,FALSE\n"
    "2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE\n"
)


def test_parses_columns_and_types() -> None:
    matches = parse_results_csv(CSV)
    assert len(matches) == 2
    m = matches[0]
    assert m == HistoricalMatch(
        date=date(2026, 6, 11),
        home_team="Mexico",
        away_team="South Africa",
        home_score=2,
        away_score=0,
        tournament="FIFA World Cup",
        neutral=False,
    )


def test_neutral_flag_parsed() -> None:
    assert parse_results_csv(CSV)[1].neutral is True


def test_rows_without_integer_scores_are_skipped() -> None:
    # Rows without a numeric score are no use to Elo: they get dropped, not smuggled in.
    csv_with_blank = CSV + "2030-01-01,A,B,,,Friendly,X,Y,FALSE\n"
    assert len(parse_results_csv(csv_with_blank)) == 2
