import re
from typing import Dict, List, Literal, Optional, Tuple

import requests
from pydantic import Field

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


# WMO weather interpretation codes — https://open-meteo.com/en/docs#weathervariables
_WEATHER_CODE_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def describe_weather_code(code: Optional[int]) -> Optional[str]:
    """Translate a WMO weather code into a human-readable description."""
    if code is None:
        return None
    return _WEATHER_CODE_DESCRIPTIONS.get(int(code), f"Code {code}")


################
# INPUT SCHEMA #
################
class WeatherToolInputSchema(BaseIOSchema):
    """
    Get current conditions and a daily/hourly forecast for a location. The
    location can be a city name (`'Brussels'`, `'New York'`, `'San Francisco, CA'`)
    or a latitude/longitude pair as `'lat,lon'`. Powered by the free Open-Meteo
    API — no key required.
    """

    location: str = Field(
        ...,
        description=(
            "City name (e.g. 'Tokyo', 'Brussels'), city + admin (`'Springfield,"
            " IL'`), or a `lat,lon` pair (e.g. `'48.8566,2.3522'`)."
        ),
    )
    units: Literal["metric", "imperial"] = Field(
        default="metric",
        description="'metric' uses Celsius and km/h; 'imperial' uses Fahrenheit and mph.",
    )
    forecast_days: int = Field(default=3, ge=1, le=16, description="Number of forecast days to return (1-16).")
    include_hourly: bool = Field(default=False, description="Whether to include the hourly forecast.")
    language: str = Field(default="en", description="Geocoder language hint (ISO 639-1).")


####################
# OUTPUT SCHEMA(S) #
####################
class WeatherCurrent(BaseIOSchema):
    """Current observed weather."""

    time: str = Field(..., description="Local timestamp of the observation (ISO 8601).")
    temperature: Optional[float] = Field(None, description="Temperature in the requested unit, or None when unavailable.")
    apparent_temperature: Optional[float] = Field(None, description="'Feels like' temperature.")
    humidity: Optional[int] = Field(None, description="Relative humidity, in percent.")
    wind_speed: Optional[float] = Field(None, description="Wind speed in km/h or mph (matches `units`).")
    wind_direction: Optional[int] = Field(None, description="Wind direction in degrees (0-360).")
    precipitation: Optional[float] = Field(None, description="Precipitation amount, in mm (metric) or inches (imperial).")
    weather_code: Optional[int] = Field(None, description="WMO weather code.")
    weather_description: Optional[str] = Field(None, description="Human-readable description of the weather code.")
    is_day: Optional[bool] = Field(None, description="True when the sun is up at the location.")


class WeatherDay(BaseIOSchema):
    """Daily forecast bucket."""

    date: str = Field(..., description="Local calendar date (YYYY-MM-DD).")
    temperature_max: Optional[float] = Field(None, description="Daily maximum temperature.")
    temperature_min: Optional[float] = Field(None, description="Daily minimum temperature.")
    apparent_temperature_max: Optional[float] = Field(None, description="Daily 'feels like' max.")
    apparent_temperature_min: Optional[float] = Field(None, description="Daily 'feels like' min.")
    precipitation_sum: Optional[float] = Field(None, description="Total precipitation for the day.")
    precipitation_probability_max: Optional[int] = Field(None, description="Max probability of precipitation, percent.")
    weather_code: Optional[int] = Field(None, description="WMO weather code (representative for the day).")
    weather_description: Optional[str] = Field(None, description="Human-readable description of the day's weather code.")
    sunrise: Optional[str] = Field(None, description="Local sunrise time (ISO 8601).")
    sunset: Optional[str] = Field(None, description="Local sunset time (ISO 8601).")
    wind_speed_max: Optional[float] = Field(None, description="Maximum wind speed.")


class WeatherHour(BaseIOSchema):
    """Hourly forecast point (only populated when include_hourly=True)."""

    time: str = Field(..., description="Local timestamp (ISO 8601).")
    temperature: Optional[float] = Field(None, description="Temperature.")
    precipitation_probability: Optional[int] = Field(None, description="Probability of precipitation, percent.")
    weather_code: Optional[int] = Field(None, description="WMO weather code.")
    weather_description: Optional[str] = Field(None, description="Human-readable description of the weather code.")


class WeatherToolOutputSchema(BaseIOSchema):
    """Output of the WeatherTool."""

    location_name: str = Field(..., description="Resolved location label.")
    country: Optional[str] = Field(None, description="Country of the resolved location.")
    latitude: float = Field(..., description="Latitude in degrees.")
    longitude: float = Field(..., description="Longitude in degrees.")
    timezone: Optional[str] = Field(None, description="IANA timezone of the location.")
    units: Dict[str, str] = Field(
        ...,
        description=(
            "Units used for the response, keyed by variable name. Example: "
            "{'temperature': '°C', 'wind_speed': 'km/h', 'precipitation': 'mm'}."
        ),
    )
    current: Optional[WeatherCurrent] = Field(None, description="Current conditions.")
    daily: List[WeatherDay] = Field(default_factory=list, description="Daily forecast buckets.")
    hourly: List[WeatherHour] = Field(default_factory=list, description="Hourly forecast points (when requested).")
    error: Optional[str] = Field(None, description="Error message when the operation failed.")


#################
# CONFIGURATION #
#################
class WeatherToolConfig(BaseToolConfig):
    """Configuration for the WeatherTool."""

    geocoding_url: str = Field(
        default="https://geocoding-api.open-meteo.com/v1/search",
        description="Open-Meteo geocoding endpoint.",
    )
    forecast_url: str = Field(
        default="https://api.open-meteo.com/v1/forecast",
        description="Open-Meteo forecast endpoint.",
    )
    user_agent: str = Field(
        default="atomic-agents-weather-tool/1.0 (+https://github.com/eigenwise/atomic-agents)",
        description="User agent for HTTP requests.",
    )
    timeout: float = Field(default=15.0, ge=1.0, le=120.0, description="HTTP request timeout in seconds.")


#####################
# MAIN TOOL & LOGIC #
#####################
class WeatherTool(BaseTool[WeatherToolInputSchema, WeatherToolOutputSchema]):
    """Weather tool backed by the free Open-Meteo APIs (geocoding + forecast)."""

    LATLON_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")

    def __init__(self, config: WeatherToolConfig = WeatherToolConfig()):
        super().__init__(config)
        self.geocoding_url = config.geocoding_url
        self.forecast_url = config.forecast_url
        self.user_agent = config.user_agent
        self.timeout = config.timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent, "Accept": "application/json"})

    @classmethod
    def parse_latlon(cls, value: str) -> Optional[Tuple[float, float]]:
        match = cls.LATLON_RE.match(value)
        if not match:
            return None
        lat, lon = float(match.group(1)), float(match.group(2))
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            return None
        return lat, lon

    def _geocode(self, query: str, language: str) -> dict:
        params = {"name": query, "count": "1", "language": language, "format": "json"}
        response = self._session.get(self.geocoding_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        results = data.get("results") or []
        if not results:
            raise ValueError(f"Location '{query}' not found.")
        return results[0]

    def _forecast(self, latitude: float, longitude: float, units: str, days: int, hourly: bool) -> dict:
        wind_unit = "kmh" if units == "metric" else "mph"
        params = {
            "latitude": str(latitude),
            "longitude": str(longitude),
            "timezone": "auto",
            "forecast_days": str(days),
            "temperature_unit": "celsius" if units == "metric" else "fahrenheit",
            "wind_speed_unit": wind_unit,
            "precipitation_unit": "mm" if units == "metric" else "inch",
            "current": ",".join(
                [
                    "temperature_2m",
                    "apparent_temperature",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "precipitation",
                    "weather_code",
                    "is_day",
                ]
            ),
            "daily": ",".join(
                [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "apparent_temperature_max",
                    "apparent_temperature_min",
                    "precipitation_sum",
                    "precipitation_probability_max",
                    "weather_code",
                    "sunrise",
                    "sunset",
                    "wind_speed_10m_max",
                ]
            ),
        }
        if hourly:
            params["hourly"] = ",".join(
                [
                    "temperature_2m",
                    "precipitation_probability",
                    "weather_code",
                ]
            )
        response = self._session.get(self.forecast_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _build_current(data: dict) -> Optional[WeatherCurrent]:
        current = data.get("current")
        if not current:
            return None
        return WeatherCurrent(
            time=current.get("time", ""),
            temperature=current.get("temperature_2m"),
            apparent_temperature=current.get("apparent_temperature"),
            humidity=current.get("relative_humidity_2m"),
            wind_speed=current.get("wind_speed_10m"),
            wind_direction=current.get("wind_direction_10m"),
            precipitation=current.get("precipitation"),
            weather_code=current.get("weather_code"),
            weather_description=describe_weather_code(current.get("weather_code")),
            is_day=bool(current.get("is_day")) if current.get("is_day") is not None else None,
        )

    @staticmethod
    def _build_daily(data: dict) -> List[WeatherDay]:
        daily = data.get("daily") or {}
        dates = daily.get("time") or []
        out: List[WeatherDay] = []
        for index, date in enumerate(dates):

            def _at(key: str):
                values = daily.get(key)
                return values[index] if isinstance(values, list) and index < len(values) else None

            code = _at("weather_code")
            out.append(
                WeatherDay(
                    date=date,
                    temperature_max=_at("temperature_2m_max"),
                    temperature_min=_at("temperature_2m_min"),
                    apparent_temperature_max=_at("apparent_temperature_max"),
                    apparent_temperature_min=_at("apparent_temperature_min"),
                    precipitation_sum=_at("precipitation_sum"),
                    precipitation_probability_max=_at("precipitation_probability_max"),
                    weather_code=code,
                    weather_description=describe_weather_code(code),
                    sunrise=_at("sunrise"),
                    sunset=_at("sunset"),
                    wind_speed_max=_at("wind_speed_10m_max"),
                )
            )
        return out

    @staticmethod
    def _build_hourly(data: dict) -> List[WeatherHour]:
        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        out: List[WeatherHour] = []
        for index, time in enumerate(times):

            def _at(key: str):
                values = hourly.get(key)
                return values[index] if isinstance(values, list) and index < len(values) else None

            code = _at("weather_code")
            out.append(
                WeatherHour(
                    time=time,
                    temperature=_at("temperature_2m"),
                    precipitation_probability=_at("precipitation_probability"),
                    weather_code=code,
                    weather_description=describe_weather_code(code),
                )
            )
        return out

    def run(self, params: WeatherToolInputSchema) -> WeatherToolOutputSchema:
        try:
            latlon = self.parse_latlon(params.location)
            if latlon:
                latitude, longitude = latlon
                location_name = f"{latitude:.4f}, {longitude:.4f}"
                country = None
            else:
                geocoded = self._geocode(params.location, params.language)
                latitude = geocoded["latitude"]
                longitude = geocoded["longitude"]
                location_name = geocoded.get("name", params.location)
                if geocoded.get("admin1"):
                    location_name = f"{location_name}, {geocoded['admin1']}"
                country = geocoded.get("country")

            data = self._forecast(latitude, longitude, params.units, params.forecast_days, params.include_hourly)
            units_map = {
                "temperature": "°C" if params.units == "metric" else "°F",
                "wind_speed": "km/h" if params.units == "metric" else "mph",
                "precipitation": "mm" if params.units == "metric" else "in",
            }

            return WeatherToolOutputSchema(
                location_name=location_name,
                country=country,
                latitude=latitude,
                longitude=longitude,
                timezone=data.get("timezone"),
                units=units_map,
                current=self._build_current(data),
                daily=self._build_daily(data),
                hourly=self._build_hourly(data) if params.include_hourly else [],
            )
        except Exception as e:
            return WeatherToolOutputSchema(
                location_name=params.location,
                country=None,
                latitude=0.0,
                longitude=0.0,
                timezone=None,
                units={},
                current=None,
                daily=[],
                hourly=[],
                error=str(e),
            )


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":  # pragma: no cover
    from rich.console import Console
    from rich.table import Table

    console = Console()
    tool = WeatherTool()

    output = tool.run(WeatherToolInputSchema(location="Brussels", units="metric", forecast_days=3))
    if output.error:
        console.print(f"[red]Error:[/red] {output.error}")
    else:
        console.rule(f"[bold cyan]{output.location_name} ({output.country}) — {output.timezone}")
        if output.current:
            console.print(
                f"Current: {output.current.temperature}{output.units['temperature']} "
                f"({output.current.weather_description}) — feels {output.current.apparent_temperature}, "
                f"humidity {output.current.humidity}%, wind {output.current.wind_speed} {output.units['wind_speed']}"
            )
        table = Table(title="Daily forecast")
        for col in ("Date", "Min", "Max", "Precip", "Conditions"):
            table.add_column(col)
        for day in output.daily:
            table.add_row(
                day.date,
                f"{day.temperature_min}{output.units['temperature']}",
                f"{day.temperature_max}{output.units['temperature']}",
                f"{day.precipitation_sum}{output.units['precipitation']}",
                day.weather_description or "",
            )
        console.print(table)
