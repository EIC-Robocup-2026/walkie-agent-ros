"""The atomic that reads the time, the date and the team, so Walkie never guesses them."""

from datetime import date, timedelta
from typing import Any

from walkie_agent.capabilities import Ctx

# English names, since strftime's %A and %B follow the process locale.
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def get_facts(ctx: Ctx) -> dict[str, Any]:
    """Read the time and the date now, and the name, country and affiliation of Walkie's team.

    Returns an object with the keys time (24-hour, such as "14:05"), today and
    tomorrow (such as "Thursday, 5 November 2026"), day_of_week, day_of_month
    (a number), team_name, team_country and team_affiliation.
    """
    now = ctx.facts.now()
    team = ctx.facts.team()
    today = now.date()
    return {
        "time": f"{now.hour:02d}:{now.minute:02d}",
        "today": spoken_date(today),
        "tomorrow": spoken_date(today + timedelta(days=1)),
        "day_of_week": WEEKDAYS[today.weekday()],
        "day_of_month": today.day,
        "team_name": team.name,
        "team_country": team.country,
        "team_affiliation": team.affiliation,
    }


def spoken_date(day: date) -> str:
    """A date as Walkie says it, such as "Thursday, 5 November 2026"."""
    return f"{WEEKDAYS[day.weekday()]}, {day.day} {MONTHS[day.month - 1]} {day.year}"
