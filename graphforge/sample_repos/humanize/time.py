"""Time humanizing functions."""

import datetime as dt
import math
from enum import Enum
from functools import total_ordering


@total_ordering
class Unit(Enum):
    MICROSECONDS = 0
    MILLISECONDS = 1
    SECONDS = 2
    MINUTES = 3
    HOURS = 4
    DAYS = 5
    MONTHS = 6
    YEARS = 7

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


def _now():
    return dt.datetime.now()


def _abs_timedelta(delta):
    if delta.days < 0:
        now = _now()
        return now - (now + delta)
    return delta


def _date_and_delta(value, *, now=None):
    if not now:
        now = _now()
    if isinstance(value, dt.datetime):
        date = value
        delta = now - value
    elif isinstance(value, dt.timedelta):
        date = now - value
        delta = value
    else:
        try:
            value = int(value)
            delta = dt.timedelta(seconds=value)
            date = now - delta
        except (ValueError, TypeError):
            return None, value
    return date, _abs_timedelta(delta)


def naturaldelta(value, months=True, minimum_unit="seconds") -> str:
    """Return a natural representation of a timedelta or number of seconds.

    Does not include tense (use naturaltime for past/future).

    Examples:
        >>> import datetime as dt
        >>> naturaldelta(dt.timedelta(seconds=90))
        'a minute'
        >>> naturaldelta(dt.timedelta(hours=2))
        '2 hours'
        >>> naturaldelta(dt.timedelta(days=400))
        'a year'
    """
    tmp = Unit[minimum_unit.upper()]
    if tmp not in (Unit.SECONDS, Unit.MILLISECONDS, Unit.MICROSECONDS):
        raise ValueError(f"Minimum unit '{minimum_unit}' not supported")
    minimum_unit = tmp

    if isinstance(value, dt.timedelta):
        delta = value
    else:
        try:
            value = int(value)
            delta = dt.timedelta(seconds=value)
        except (ValueError, TypeError):
            return value

    seconds = abs(delta.seconds)
    days = abs(delta.days)
    years = days // 365
    days = days % 365
    months_count = int(days // 30.5)

    if not years and days < 1:
        if seconds == 0:
            return "a moment"
        elif seconds == 1:
            return "a second"
        elif seconds < 60:
            return f"{seconds} seconds" if seconds > 1 else "a second"
        elif 60 <= seconds < 120:
            return "a minute"
        elif 120 <= seconds < 3600:
            minutes = seconds // 60
            return f"{minutes} minutes"
        elif 3600 <= seconds < 7200:
            return "an hour"
        else:
            hours = seconds // 3600
            return f"{hours} hours"
    elif years == 0:
        if days == 1:
            return "a day"
        if not months or not months_count:
            return f"{days} days"
        elif months_count == 1:
            return "a month"
        return f"{months_count} months"
    elif years == 1:
        if not months_count and not days:
            return "a year"
        elif not months_count:
            return f"1 year, {days} days" if days > 1 else "1 year, a day"
        elif months_count == 1:
            return "1 year, 1 month"
        return f"1 year, {months_count} months"
    return f"{years} years"


def naturaltime(value, future=False, months=True, minimum_unit="seconds", when=None) -> str:
    """Return a natural representation of a time relative to now.

    Examples:
        >>> import datetime as dt
        >>> naturaltime(dt.timedelta(seconds=30))
        '30 seconds ago'
        >>> naturaltime(dt.timedelta(hours=1), future=True)
        'an hour from now'
    """
    now = when or _now()
    date, delta = _date_and_delta(value, now=now)
    if date is None:
        return value
    if isinstance(value, (dt.datetime, dt.timedelta)):
        future = date > now
    ago = "%s from now" if future else "%s ago"
    delta_str = naturaldelta(delta, months, minimum_unit)
    if delta_str == "a moment":
        return "now"
    return ago % delta_str


def naturalday(value, format="%b %d") -> str:
    """Return 'today', 'tomorrow', 'yesterday', or a formatted date string.

    Examples:
        >>> import datetime as dt
        >>> naturalday(dt.date.today())
        'today'
    """
    try:
        value = dt.date(value.year, value.month, value.day)
    except (AttributeError, OverflowError, ValueError):
        return value
    delta = value - dt.date.today()
    if delta.days == 0:
        return "today"
    elif delta.days == 1:
        return "tomorrow"
    elif delta.days == -1:
        return "yesterday"
    return value.strftime(format)


def naturaldate(value) -> str:
    """Like naturalday, but appends year for dates more than ~5 months away."""
    try:
        value = dt.date(value.year, value.month, value.day)
    except (AttributeError, OverflowError, ValueError):
        return value
    delta = _abs_timedelta(value - dt.date.today())
    if delta.days >= 5 * 365 / 12:
        return naturalday(value, "%b %d %Y")
    return naturalday(value)


def precisedelta(value, minimum_unit="seconds", suppress=(), format="%0.2f") -> str:
    """Return a precise, human-readable representation of a timedelta.

    Examples:
        >>> import datetime as dt
        >>> precisedelta(dt.timedelta(seconds=3633, days=2))
        '2 days and 1 hour and 33 seconds'
    """
    date, delta = _date_and_delta(value)
    if date is None:
        return value

    suppress_units = {Unit[s.upper()] for s in suppress}
    min_unit = Unit[minimum_unit.upper()]

    days = delta.days
    secs = delta.seconds

    years, days = divmod(days, 365)
    months_count = int(days // 30.5)
    days = days % 30

    hours, secs = divmod(secs, 3600)
    minutes, secs = divmod(secs, 60)

    parts = []
    for count, singular, plural in [
        (years,         "year",   "years"),
        (months_count,  "month",  "months"),
        (days,          "day",    "days"),
        (hours,         "hour",   "hours"),
        (minutes,       "minute", "minutes"),
        (secs,          "second", "seconds"),
    ]:
        if count > 0:
            label = singular if count == 1 else plural
            parts.append(f"{count} {label}")

    if not parts:
        return "0 seconds"
    if len(parts) == 1:
        return parts[0]
    return " and ".join(parts)
