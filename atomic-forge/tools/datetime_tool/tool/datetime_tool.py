from datetime import datetime, timedelta, timezone
from typing import Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


################
# INPUT SCHEMA #
################
class DateTimeToolInputSchema(BaseIOSchema):
    """
    Tool for date and time operations: getting the current time in any timezone,
    parsing datetime strings, converting between timezones, shifting a datetime by a
    duration, and computing the difference between two datetimes. Use this whenever
    you need to know "what is today's date", convert a meeting time across regions,
    or do any temporal arithmetic.
    """

    operation: Literal["now", "parse", "convert", "shift", "diff"] = Field(
        ...,
        description=(
            "The operation to perform. 'now' returns the current time. 'parse' parses"
            " an input datetime string. 'convert' converts a datetime from one"
            " timezone to another. 'shift' adds a duration to a datetime. 'diff'"
            " computes the time difference between two datetimes."
        ),
    )
    timezone: str = Field(
        default="UTC",
        description=(
            "IANA timezone name to interpret the input/result in (e.g. 'UTC',"
            " 'America/New_York', 'Europe/Brussels', 'Asia/Tokyo'). For 'convert'"
            " this is the source timezone of `input`."
        ),
    )
    target_timezone: Optional[str] = Field(
        default=None,
        description="Target IANA timezone for the 'convert' operation.",
    )
    input: Optional[str] = Field(
        default=None,
        description=(
            "Datetime string for 'parse', 'convert', and 'shift' operations. ISO 8601"
            " is preferred (e.g. '2026-04-29T15:30:00'). If the string has no"
            " timezone info, `timezone` is used."
        ),
    )
    input_2: Optional[str] = Field(
        default=None,
        description="Second datetime string for the 'diff' operation (computes input_2 - input).",
    )
    days: int = Field(default=0, description="Calendar days to add for 'shift', before elapsed hours/minutes/seconds.")
    hours: int = Field(default=0, description="Elapsed hours to add for 'shift' (negative to subtract).")
    minutes: int = Field(default=0, description="Elapsed minutes to add for 'shift' (negative to subtract).")
    seconds: int = Field(default=0, description="Elapsed seconds to add for 'shift' (negative to subtract).")


#################
# OUTPUT SCHEMA #
#################
class DateTimeToolOutputSchema(BaseIOSchema):
    """Output of the DateTimeTool. `error` is populated only when the operation fails."""

    operation: str = Field(..., description="The operation that was performed.")
    iso: Optional[str] = Field(None, description="ISO 8601 representation of the resulting datetime.")
    timezone: Optional[str] = Field(None, description="IANA timezone of the resulting datetime.")
    unix_timestamp: Optional[float] = Field(None, description="POSIX/Unix timestamp (seconds since epoch).")
    year: Optional[int] = Field(None, description="Year component of the resulting datetime.")
    month: Optional[int] = Field(None, description="Month component (1-12).")
    day: Optional[int] = Field(None, description="Day-of-month component (1-31).")
    hour: Optional[int] = Field(None, description="Hour component (0-23).")
    minute: Optional[int] = Field(None, description="Minute component (0-59).")
    second: Optional[int] = Field(None, description="Second component (0-59).")
    weekday: Optional[str] = Field(None, description="Day name, e.g. 'Monday'.")
    human: Optional[str] = Field(None, description="Human-readable formatted datetime, e.g. 'Wed, 29 Apr 2026 15:30 UTC'.")
    diff_seconds: Optional[float] = Field(None, description="Total seconds between input_2 and input (for 'diff').")
    diff_human: Optional[str] = Field(None, description="Human-readable difference, e.g. '2 days, 3 hours, 15 minutes'.")
    error: Optional[str] = Field(None, description="Error message if the operation failed.")


#################
# CONFIGURATION #
#################
class DateTimeToolConfig(BaseToolConfig):
    """Configuration for the DateTimeTool. Currently has no tunable settings beyond title/description overrides."""


#####################
# MAIN TOOL & LOGIC #
#####################
class DateTimeTool(BaseTool[DateTimeToolInputSchema, DateTimeToolOutputSchema]):
    """Date and time operations: now, parse, convert, shift, diff."""

    def __init__(self, config: DateTimeToolConfig = DateTimeToolConfig()):
        super().__init__(config)

    @staticmethod
    def _zone(name: str) -> ZoneInfo:
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError as e:
            raise ValueError(f"Unknown timezone '{name}'. Use an IANA name like 'UTC' or 'America/New_York'.") from e

    @classmethod
    def _parse(cls, value: str, default_tz: str) -> datetime:
        """
        Parse an ISO 8601 datetime string. Trailing 'Z' is treated as UTC. If the
        parsed value is naive (no tzinfo), `default_tz` is attached.
        """
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as e:
            raise ValueError(f"Could not parse datetime '{value}': {e}") from e
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=cls._zone(default_tz))
        return dt

    @staticmethod
    def _humanize_delta(delta: timedelta) -> str:
        total = int(delta.total_seconds())
        sign = "-" if total < 0 else ""
        total = abs(total)
        days, remainder = divmod(total, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts = []
        if days:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds or not parts:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
        return sign + ", ".join(parts)

    def _build(self, dt: datetime, operation: str) -> DateTimeToolOutputSchema:
        tzname = str(dt.tzinfo) if dt.tzinfo and not isinstance(dt.tzinfo, timezone) else (dt.tzname() or "UTC")
        return DateTimeToolOutputSchema(
            operation=operation,
            iso=dt.isoformat(),
            timezone=tzname,
            unix_timestamp=dt.timestamp(),
            year=dt.year,
            month=dt.month,
            day=dt.day,
            hour=dt.hour,
            minute=dt.minute,
            second=dt.second,
            weekday=dt.strftime("%A"),
            human=dt.strftime("%a, %d %b %Y %H:%M %Z").strip(),
        )

    def run(self, params: DateTimeToolInputSchema) -> DateTimeToolOutputSchema:
        try:
            if params.operation == "now":
                dt = datetime.now(self._zone(params.timezone))
                return self._build(dt, "now")

            if params.operation == "parse":
                if not params.input:
                    raise ValueError("'parse' operation requires `input`.")
                dt = self._parse(params.input, params.timezone)
                return self._build(dt, "parse")

            if params.operation == "convert":
                if not params.input or not params.target_timezone:
                    raise ValueError("'convert' operation requires `input` and `target_timezone`.")
                dt = self._parse(params.input, params.timezone).astimezone(self._zone(params.target_timezone))
                return self._build(dt, "convert")

            if params.operation == "shift":
                if not params.input:
                    raise ValueError("'shift' operation requires `input`.")
                dt = self._parse(params.input, params.timezone)
                # Calendar days retain the local clock time. Do not add a zero
                # timedelta: datetime arithmetic resets fold during an overlap.
                if params.days:
                    dt += timedelta(days=params.days)
                # Sub-day components represent elapsed time, including across DST.
                # The round trip also normalizes a calendar result inside a gap.
                dt = (
                    dt.astimezone(timezone.utc) + timedelta(hours=params.hours, minutes=params.minutes, seconds=params.seconds)
                ).astimezone(dt.tzinfo)
                return self._build(dt, "shift")

            if params.operation == "diff":
                if not params.input or not params.input_2:
                    raise ValueError("'diff' operation requires `input` and `input_2`.")
                # Subtract in UTC: Python subtracts datetimes that share a tzinfo by wall clock,
                # which ignores a DST change between them.
                a = self._parse(params.input, params.timezone).astimezone(timezone.utc)
                b = self._parse(params.input_2, params.timezone).astimezone(timezone.utc)
                delta = b - a
                return DateTimeToolOutputSchema(
                    operation="diff",
                    diff_seconds=delta.total_seconds(),
                    diff_human=self._humanize_delta(delta),
                )

            raise ValueError(f"Unknown operation '{params.operation}'.")
        except Exception as e:
            return DateTimeToolOutputSchema(operation=params.operation, error=str(e))


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":  # pragma: no cover
    from rich.console import Console

    console = Console()
    tool = DateTimeTool()

    console.rule("[bold cyan]now in Asia/Tokyo")
    console.print(tool.run(DateTimeToolInputSchema(operation="now", timezone="Asia/Tokyo")))

    console.rule("[bold cyan]parse 2026-04-29T15:30 (Europe/Brussels)")
    console.print(
        tool.run(DateTimeToolInputSchema(operation="parse", input="2026-04-29T15:30:00", timezone="Europe/Brussels"))
    )

    console.rule("[bold cyan]convert Europe/Brussels -> America/New_York")
    console.print(
        tool.run(
            DateTimeToolInputSchema(
                operation="convert",
                input="2026-04-29T15:30:00",
                timezone="Europe/Brussels",
                target_timezone="America/New_York",
            )
        )
    )

    console.rule("[bold cyan]shift +3 days, -2 hours")
    console.print(
        tool.run(
            DateTimeToolInputSchema(
                operation="shift",
                input="2026-04-29T15:30:00",
                timezone="UTC",
                days=3,
                hours=-2,
            )
        )
    )

    console.rule("[bold cyan]diff between two datetimes")
    console.print(
        tool.run(
            DateTimeToolInputSchema(
                operation="diff",
                input="2026-04-29T15:30:00Z",
                input_2="2026-05-02T18:45:00Z",
            )
        )
    )
