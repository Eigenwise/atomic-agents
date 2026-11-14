from datetime import datetime, timedelta

import pytest

from tool.datetime_tool import DateTimeTool, DateTimeToolInputSchema


@pytest.mark.parametrize(
    "zone,value",
    [
        ("America/New_York", "2026-03-08T02:00:00"),
        ("America/New_York", "2026-03-08T02:59:59.999999"),
        ("Australia/Lord_Howe", "2026-10-04T02:15:00"),
        ("Pacific/Apia", "2011-12-30T12:00:00"),
    ],
)
def test_parse_rejects_local_times_that_never_occurred(zone, value):
    output = DateTimeTool().run(DateTimeToolInputSchema(operation="parse", input=value, timezone=zone))

    assert output.error is not None
    assert "does not exist" in output.error
    assert zone in output.error
    assert value in output.error
    assert output.iso is None
    assert output.unix_timestamp is None


@pytest.mark.parametrize("operation", ["convert", "shift", "diff"])
def test_nonexistent_input_is_rejected_before_other_operations(operation):
    output = DateTimeTool().run(
        DateTimeToolInputSchema(
            operation=operation,
            input="2026-03-08T02:30:00",
            input_2="2026-03-08T04:00:00",
            timezone="America/New_York",
            target_timezone="UTC",
            hours=1,
        )
    )

    assert output.error is not None
    assert "does not exist" in output.error
    assert output.iso is None
    assert output.diff_seconds is None


def test_diff_rejects_a_nonexistent_second_input():
    output = DateTimeTool().run(
        DateTimeToolInputSchema(
            operation="diff",
            input="2026-03-08T01:30:00",
            input_2="2026-03-08T02:30:00",
            timezone="America/New_York",
        )
    )

    assert output.error is not None
    assert "does not exist" in output.error
    assert output.diff_seconds is None


@pytest.mark.parametrize("value", ["2026-03-08T01:59:59.999999", "2026-03-08T03:00:00"])
def test_both_sides_of_the_gap_remain_valid(value):
    output = DateTimeTool().run(DateTimeToolInputSchema(operation="parse", input=value, timezone="America/New_York"))

    assert output.error is None
    assert datetime.fromisoformat(output.iso).replace(tzinfo=None) == datetime.fromisoformat(value)


@pytest.mark.parametrize(
    "value,expected_offset",
    [
        ("2026-11-01T01:30:00", timedelta(hours=-4)),
        ("2026-11-01T01:30:00-05:00", timedelta(hours=-5)),
        ("2026-03-08T02:30:00-05:00", timedelta(hours=-5)),
    ],
)
def test_overlap_default_and_explicit_offsets_keep_their_meaning(value, expected_offset):
    output = DateTimeTool().run(DateTimeToolInputSchema(operation="parse", input=value, timezone="America/New_York"))

    assert output.error is None
    parsed = datetime.fromisoformat(output.iso)
    assert parsed.utcoffset() == expected_offset
    assert output.unix_timestamp == parsed.timestamp()


@pytest.mark.parametrize(
    "zone,value",
    [
        ("Asia/Tokyo", "0001-01-01T00:00:00"),
        ("America/New_York", "9999-12-31T23:59:59"),
        ("UTC", "2026-03-08T02:30:00"),
    ],
)
def test_ordinary_offsets_do_not_require_a_representable_utc_round_trip(zone, value):
    output = DateTimeTool().run(DateTimeToolInputSchema(operation="parse", input=value, timezone=zone))

    assert output.error is None
    assert datetime.fromisoformat(output.iso).replace(tzinfo=None) == datetime.fromisoformat(value)
