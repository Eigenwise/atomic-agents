import pytest

from tool.datetime_tool import (
    DateTimeTool,
    DateTimeToolInputSchema,
    DateTimeToolOutputSchema,
)


def test_now_default_utc():
    tool = DateTimeTool()
    out = tool.run(DateTimeToolInputSchema(operation="now"))
    assert isinstance(out, DateTimeToolOutputSchema)
    assert out.operation == "now"
    assert out.iso is not None
    assert out.error is None
    assert out.weekday in {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}


def test_now_in_tokyo():
    tool = DateTimeTool()
    out = tool.run(DateTimeToolInputSchema(operation="now", timezone="Asia/Tokyo"))
    assert out.error is None
    assert out.timezone == "Asia/Tokyo"
    assert "+09:00" in out.iso


def test_parse_iso_with_timezone():
    tool = DateTimeTool()
    out = tool.run(DateTimeToolInputSchema(operation="parse", input="2026-04-29T15:30:00Z"))
    assert out.error is None
    assert (out.year, out.month, out.day, out.hour, out.minute) == (2026, 4, 29, 15, 30)
    assert out.iso.startswith("2026-04-29T15:30:00")


def test_parse_naive_uses_default_tz():
    tool = DateTimeTool()
    out = tool.run(DateTimeToolInputSchema(operation="parse", input="2026-04-29T15:30:00", timezone="Europe/Brussels"))
    assert out.error is None
    assert out.timezone == "Europe/Brussels"


def test_convert_brussels_to_new_york():
    tool = DateTimeTool()
    out = tool.run(
        DateTimeToolInputSchema(
            operation="convert",
            input="2026-04-29T15:30:00",
            timezone="Europe/Brussels",
            target_timezone="America/New_York",
        )
    )
    assert out.error is None
    assert out.timezone == "America/New_York"
    assert (out.year, out.month, out.day) == (2026, 4, 29)
    assert (out.hour, out.minute) == (9, 30)


def test_shift_forward_and_backward():
    tool = DateTimeTool()
    out = tool.run(
        DateTimeToolInputSchema(
            operation="shift",
            input="2026-04-29T00:00:00Z",
            days=3,
            hours=-2,
            minutes=30,
        )
    )
    assert out.error is None
    assert out.year == 2026
    assert out.month == 5
    assert out.day == 1
    assert out.hour == 22
    assert out.minute == 30


def test_diff_positive():
    tool = DateTimeTool()
    out = tool.run(
        DateTimeToolInputSchema(
            operation="diff",
            input="2026-04-29T00:00:00Z",
            input_2="2026-04-30T01:30:00Z",
        )
    )
    assert out.error is None
    assert out.diff_seconds == 25 * 3600 + 30 * 60
    assert "1 day" in out.diff_human and "1 hour" in out.diff_human and "30 minute" in out.diff_human


def test_diff_negative_keeps_sign():
    tool = DateTimeTool()
    out = tool.run(
        DateTimeToolInputSchema(
            operation="diff",
            input="2026-04-30T00:00:00Z",
            input_2="2026-04-29T00:00:00Z",
        )
    )
    assert out.error is None
    assert out.diff_seconds == -86400.0
    assert out.diff_human.startswith("-")


def test_diff_counts_elapsed_time_across_a_dst_change():
    tool = DateTimeTool()
    # Clocks in New York go forward an hour on 2026-03-08, and in Brussels back an hour on 2026-10-25.
    spring = tool.run(
        DateTimeToolInputSchema(
            operation="diff",
            input="2026-03-08T00:00:00",
            input_2="2026-03-09T00:00:00",
            timezone="America/New_York",
        )
    )
    autumn = tool.run(
        DateTimeToolInputSchema(
            operation="diff",
            input="2026-10-25T00:00:00",
            input_2="2026-10-26T00:00:00",
            timezone="Europe/Brussels",
        )
    )
    assert spring.error is None and autumn.error is None
    assert spring.diff_seconds == 23 * 3600
    assert spring.diff_human == "23 hours"
    assert autumn.diff_seconds == 25 * 3600
    assert autumn.diff_human == "1 day, 1 hour"


def test_unknown_timezone_returns_error():
    tool = DateTimeTool()
    out = tool.run(DateTimeToolInputSchema(operation="now", timezone="Mars/Olympus"))
    assert out.error is not None
    assert "Unknown timezone" in out.error


def test_missing_input_for_parse_returns_error():
    tool = DateTimeTool()
    out = tool.run(DateTimeToolInputSchema(operation="parse"))
    assert out.error is not None
    assert "input" in out.error


def test_missing_target_timezone_for_convert():
    tool = DateTimeTool()
    out = tool.run(DateTimeToolInputSchema(operation="convert", input="2026-04-29T15:30:00Z"))
    assert out.error is not None
    assert "target_timezone" in out.error


def test_unparseable_input():
    tool = DateTimeTool()
    out = tool.run(DateTimeToolInputSchema(operation="parse", input="not-a-date"))
    assert out.error is not None
    assert "Could not parse" in out.error


@pytest.mark.parametrize(
    "source,zone,seconds,expected",
    [
        ("2026-03-08T00:00:00", "America/New_York", 86400, "2026-03-09T01:00:00-04:00"),
        ("2026-11-01T00:00:00", "America/New_York", 86400, "2026-11-01T23:00:00-05:00"),
        ("2026-03-08T01:30:00", "America/New_York", 3600, "2026-03-08T03:30:00-04:00"),
        ("2026-11-01T02:30:00", "America/New_York", -7200, "2026-11-01T01:30:00-04:00"),
        ("2026-10-25T00:00:00", "Europe/Brussels", 86400, "2026-10-25T23:00:00+01:00"),
    ],
)
def test_shift_seconds_uses_elapsed_time(source, zone, seconds, expected):
    tool = DateTimeTool()
    out = tool.run(DateTimeToolInputSchema(operation="shift", input=source, timezone=zone, seconds=seconds))
    assert out.error is None
    assert out.iso == expected
    elapsed = tool.run(DateTimeToolInputSchema(operation="diff", input=source, input_2=out.iso, timezone=zone))
    assert elapsed.diff_seconds == seconds


def test_shift_calendar_day_differs_from_24_elapsed_hours():
    tool = DateTimeTool()
    source = "2026-03-08T00:00:00"
    day = tool.run(DateTimeToolInputSchema(operation="shift", input=source, timezone="America/New_York", days=1))
    hours = tool.run(DateTimeToolInputSchema(operation="shift", input=source, timezone="America/New_York", hours=24))
    assert day.iso == "2026-03-09T00:00:00-04:00"
    assert hours.iso == "2026-03-09T01:00:00-04:00"


def test_shift_applies_calendar_days_before_elapsed_minutes():
    out = DateTimeTool().run(
        DateTimeToolInputSchema(
            operation="shift", input="2026-03-07T01:30:00", timezone="America/New_York", days=1, minutes=60
        )
    )
    assert out.error is None
    assert out.iso == "2026-03-08T03:30:00-04:00"


def test_zero_shift_preserves_second_occurrence_of_ambiguous_time():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tool = DateTimeTool()
    # ISO offsets identify the second 01:30 unambiguously when converted to New York.
    original = datetime.fromisoformat("2026-11-01T01:30:00-05:00").astimezone(ZoneInfo("America/New_York"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tool, "_parse", lambda value, zone: original)
        out = tool.run(DateTimeToolInputSchema(operation="shift", input="unused", timezone="America/New_York"))
    assert out.iso == original.isoformat()
    assert out.unix_timestamp == original.timestamp()


def test_calendar_shift_normalizes_a_nonexistent_local_time():
    out = DateTimeTool().run(
        DateTimeToolInputSchema(operation="shift", input="2026-03-07T02:30:00", timezone="America/New_York", days=1)
    )
    assert out.iso == "2026-03-08T03:30:00-04:00"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
