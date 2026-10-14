"""Seeded generator for the synthetic marketing data.

It simulates an online shop's paid channels day by day and writes three tables:

``channel_daily.csv``
    One row per day and paid channel: spend, impressions, clicks, and the purchases
    and revenue the channel's spend *caused* (including carryover from earlier days,
    so a day with no spend can still show some). Only a generator can know these;
    they are the truth an attribution or mix model is judged against.
``ga4_sessions.csv``
    One row per website session, shaped like a Google Analytics 4 export flattened to
    sessions: traffic source, medium and campaign, default channel group, device, and
    the purchase made in the session, if any. Every purchase is credited to the last
    session before it, so this table shows last-click attribution, which differs from
    the causal split above in the way it does in practice: search and direct get credit
    for purchases display and video caused.
``truth_daily.csv``
    The components each day's purchases were drawn from: each channel's adstocked spend
    and expected purchases, the baseline demand, and repeat purchases.

``truth.json`` records the configuration, each channel's true response parameters and
the rates the run actually produced.

How a day is made:

- **Demand** is a trend times yearly seasonality (peaking in early December), a weekly
  pattern, and a boost from Black Friday to Cyber Monday.
- **Spend** per channel follows a quarterly budget level, partly follows demand, has
  daily noise, and for flighted channels switches on and off for weeks at a time. It is
  rescaled so its mean over the period is the channel's ``daily_spend``.
- **Clicks and impressions** follow spend through the channel's cost per click and
  click-through rate.
- **Purchases a channel causes** are Poisson with mean ``beta * hill(adstock(spend))``:
  geometric adstock (normalised, so it stays in dollars per day) then a Hill curve.
  ``beta`` is set so the channel's purchases per click over the period match its
  conversion rate.
- **Baseline purchases** (new customers no paid channel caused) are Poisson with mean
  ``baseline_daily_conversions * demand``.
- **Repeat purchases**: each purchase is followed by another from the same customer with
  probability ``repeat_probability``, after a gap of whole days, geometric with mean
  ``repeat_gap_days``.
- **Order values** are log-normal with mean ``aov``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from marketing_roi.channels import CHANNELS, Channel

# Unpaid traffic: (source, medium, campaign, default channel group, mean sessions a day
# at a demand of 1).
UNPAID: dict[str, tuple[str, str, str, str, float]] = {
    "direct": ("(direct)", "(none)", "(direct)", "Direct", 350.0),
    "organic_search": ("google", "organic", "(organic)", "Organic Search", 450.0),
    "referral": ("partner-blog.example", "referral", "(referral)", "Referral", 80.0),
    "email": ("newsletter", "email", "weekly-newsletter", "Email", 120.0),
}

# Where the last click before a purchase falls, by what caused the purchase. A paid
# channel's own share is its `last_click_share`; the rest of its purchases go here.
STOLEN_FROM_PAID = {"brand_search": 0.25, "organic_search": 0.40, "direct": 0.35}
BASELINE_LAST_CLICK = {
    "organic_search": 0.45,
    "direct": 0.35,
    "referral": 0.10,
    "brand_search": 0.10,
}
REPEAT_LAST_CLICK = {"direct": 0.40, "email": 0.40, "organic_search": 0.20}


def stolen_from(ch: Channel) -> dict[str, float]:
    """Where the last click falls for the purchases `ch` causes but does not get: not one
    of its own sessions, so paid search's share leaves out its brand campaign."""
    options = {
        k: v
        for k, v in STOLEN_FROM_PAID.items()
        if not (k == "brand_search" and ch.name == "paid_search")
    }
    total = sum(options.values())
    return {k: v / total for k, v in options.items()}


# Monday to Sunday; mean 1.
WEEKLY = np.array([1.00, 1.02, 1.03, 1.00, 0.96, 0.94, 1.05])
DEVICES = ("mobile", "desktop", "tablet")
DEVICE_SHARES = (0.58, 0.37, 0.05)

CHANNEL_DAILY_COLUMNS = (
    "date",
    "channel",
    "spend",
    "impressions",
    "clicks",
    "conversions",
    "revenue",
)
SESSION_COLUMNS = (
    "session_id",
    "session_start",
    "user_pseudo_id",
    "source",
    "medium",
    "campaign",
    "default_channel_group",
    "device_category",
    "engaged_session",
    "transaction_id",
    "purchase_revenue",
)
TRUTH_DAILY_COLUMNS = (
    "date",
    "component",
    "adstocked_spend",
    "expected_conversions",
    "conversions",
    "revenue",
)


def _check_number(
    name: str,
    value: object,
    low: float,
    high: float,
    *,
    integer: bool = False,
    low_open: bool = False,
) -> None:
    kinds = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds):
        kind = "an integer" if integer else "a number"
        raise TypeError(f"{name} must be {kind}, not {type(value).__name__}")
    if not math.isfinite(value) or value > high or value < low or (low_open and value == low):
        bound = f"({low}, {high}]" if low_open else f"[{low}, {high}]"
        raise ValueError(f"{name} must be in {bound}, got {value!r}")


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    start: dt.date = dt.date(2024, 1, 1)
    days: int = 731
    baseline_daily_conversions: float = 14.0
    trend_per_year: float = 0.08
    seasonal_amplitude: float = 0.15
    seasonal_peak_day: int = 340
    """Day of the year demand peaks (340 is early December)."""
    holiday_boost: float = 0.6
    """Extra demand from Black Friday to Cyber Monday, as a share of the day's demand."""
    aov: float = 90.0
    aov_sigma: float = 0.5
    repeat_probability: float = 0.35
    repeat_gap_days: float = 75.0
    landing_rate: float = 0.88
    """Share of paid clicks that become a recorded session."""
    engaged_rate: float = 0.55
    """Share of sessions without a purchase that GA4 would count as engaged."""
    visits_before_purchase: float = 1.5
    """Mean number of earlier visits a customer makes before each purchase (Poisson)."""
    visit_window_days: int = 14
    """Those visits fall 1 to this many days before the purchase."""
    channels: tuple[Channel, ...] = field(default=CHANNELS)

    def __post_init__(self) -> None:
        _check_number("seed", self.seed, 0, 2**63 - 1, integer=True)
        if not isinstance(self.start, dt.date) or isinstance(self.start, dt.datetime):
            raise TypeError(f"start must be a date, not {type(self.start).__name__}")
        _check_number("days", self.days, 1, 20 * 366, integer=True)
        _check_number("baseline_daily_conversions", self.baseline_daily_conversions, 0, 1e5)
        _check_number("trend_per_year", self.trend_per_year, -0.5, 1.0)
        _check_number("seasonal_amplitude", self.seasonal_amplitude, 0, 0.9)
        _check_number("seasonal_peak_day", self.seasonal_peak_day, 1, 366, integer=True)
        _check_number("holiday_boost", self.holiday_boost, 0, 5)
        _check_number("aov", self.aov, 0, 1e5, low_open=True)
        _check_number("aov_sigma", self.aov_sigma, 0, 2)
        _check_number("repeat_probability", self.repeat_probability, 0, 0.95)
        _check_number("repeat_gap_days", self.repeat_gap_days, 1, 3650)
        _check_number("landing_rate", self.landing_rate, 0, 1, low_open=True)
        _check_number("engaged_rate", self.engaged_rate, 0, 1)
        _check_number("visits_before_purchase", self.visits_before_purchase, 0, 20)
        _check_number("visit_window_days", self.visit_window_days, 1, 365, integer=True)
        if not self.channels or not all(isinstance(c, Channel) for c in self.channels):
            raise TypeError("channels must be a non-empty tuple of Channel")
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names):
            raise ValueError(f"channel names must be distinct, got {names}")
        if any(n in UNPAID or n == "brand_search" for n in names):
            raise ValueError("a channel name clashes with an unpaid source")
        if "paid_search" not in names or "brand" not in _channel(self, "paid_search").campaigns:
            raise ValueError("a paid_search channel with a 'brand' campaign is required")


def _channel(config: GeneratorConfig, name: str) -> Channel:
    return next(c for c in config.channels if c.name == name)


@dataclass
class Dataset:
    """A generated run. Money is held in integer cents so every total is exact."""

    config: GeneratorConfig
    dates: np.ndarray  # datetime64[D], one per day
    demand: np.ndarray
    channel_daily: dict[str, np.ndarray]
    truth_daily: dict[str, np.ndarray]
    sessions: dict[str, np.ndarray]
    session_keys: list[tuple[str, str, str, str]]
    """(source, medium, campaign, default channel group) per session key index."""
    betas: dict[str, float]
    fallback_purchases: int
    """Purchases whose drawn last-click source had no free session that day, credited to
    a direct session instead."""
    added_direct_sessions: int
    """Direct sessions added because direct had too few for the purchases sent to it."""


def adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Normalised geometric adstock: a[t] = (1 - decay) * spend[t] + decay * a[t - 1].

    Starts from a[-1] = spend[0], as if the first day's spend had run for ever, so a
    constant spend gives a constant adstock equal to it.
    """
    out = np.empty(len(spend), dtype=float)
    carry = float(spend[0]) if len(spend) else 0.0
    for t, s in enumerate(spend):
        carry = (1.0 - decay) * float(s) + decay * carry
        out[t] = carry
    return out


def hill(x: np.ndarray, half_saturation: float, shape: float) -> np.ndarray:
    """x^shape / (x^shape + half_saturation^shape); 0 at 0, 1/2 at half_saturation."""
    x = np.asarray(x, dtype=float)
    xs = np.power(x, shape)
    return xs / (xs + half_saturation**shape)


def black_friday(year: int) -> dt.date:
    """The day after the fourth Thursday of November."""
    first = dt.date(year, 11, 1)
    first_thursday = first + dt.timedelta(days=(3 - first.weekday()) % 7)
    return first_thursday + dt.timedelta(days=22)


def demand_curve(config: GeneratorConfig, dates: list[dt.date]) -> np.ndarray:
    t = np.arange(len(dates), dtype=float)
    doy = np.array([d.timetuple().tm_yday for d in dates], dtype=float)
    dow = np.array([d.weekday() for d in dates])
    seasonal = 1 + config.seasonal_amplitude * np.cos(
        2 * np.pi * (doy - config.seasonal_peak_day) / 365.25
    )
    trend = 1 + config.trend_per_year * t / 365.25
    holiday = np.ones(len(dates))
    for i, d in enumerate(dates):
        bf = black_friday(d.year)
        if bf <= d <= bf + dt.timedelta(days=3):
            holiday[i] += config.holiday_boost
    return trend * seasonal * WEEKLY[dow] * holiday


def _flights(rng: np.random.Generator, n: int) -> np.ndarray:
    on = np.zeros(n, dtype=bool)
    t, running = 0, True
    while t < n:
        length = int(rng.integers(21, 57) if running else rng.integers(7, 29))
        on[t : t + length] = running
        t += length
        running = not running
    return on


def _spend_cents(
    rng: np.random.Generator, ch: Channel, dates: list[dt.date], demand: np.ndarray
) -> np.ndarray:
    n = len(dates)
    quarter = np.array([(d.year - dates[0].year) * 4 + (d.month - 1) // 3 for d in dates])
    levels = rng.uniform(0.6, 1.4, size=quarter.max() + 1)[quarter]
    spend = levels * (1 + 0.5 * (demand / demand.mean() - 1)) * rng.lognormal(0, 0.15, n)
    if ch.flighted:
        spend = spend * _flights(rng, n)
    spend = spend * (ch.daily_spend / spend.mean())
    return np.round(spend * 100).astype(np.int64)


def _order_values_cents(rng: np.random.Generator, config: GeneratorConfig, n: int) -> np.ndarray:
    s = config.aov_sigma
    values = config.aov * rng.lognormal(-s * s / 2, s, n)
    return np.maximum(np.round(values * 100), 1).astype(np.int64)


def repeat_gaps(rng: np.random.Generator, mean_days: float, n: int) -> np.ndarray:
    """Whole days between a purchase and the next: geometric on 1, 2, ... with mean
    `mean_days`."""
    return rng.geometric(1.0 / mean_days, n)


def _draw(rng: np.random.Generator, options: dict[str, float], n: int) -> np.ndarray:
    names = list(options)
    return np.array(names, dtype=object)[rng.choice(len(names), size=n, p=list(options.values()))]


def generate(config: GeneratorConfig | None = None) -> Dataset:
    config = config or GeneratorConfig()
    rng = np.random.default_rng(config.seed)
    n = config.days
    dates = [config.start + dt.timedelta(days=i) for i in range(n)]
    demand = demand_curve(config, dates)
    chans = config.channels
    nc = len(chans)

    # Paid channels: spend, funnel and the purchases each one causes.
    spend = np.stack([_spend_cents(rng, ch, dates, demand) for ch in chans])
    impressions = np.empty_like(spend)
    clicks = np.empty_like(spend)
    adstocked = np.empty(spend.shape)
    expected = np.empty(spend.shape)
    betas: dict[str, float] = {}
    for i, ch in enumerate(chans):
        dollars = spend[i] / 100
        impressions[i] = rng.poisson(dollars / (ch.cpc * ch.ctr))
        clicks[i] = rng.binomial(impressions[i], ch.ctr)
        adstocked[i] = adstock(dollars, ch.decay)
        response = hill(adstocked[i], ch.half_saturation * ch.daily_spend, ch.hill_shape)
        beta = float((dollars / ch.cpc).sum() * ch.cvr / response.sum())
        betas[ch.name] = beta
        expected[i] = beta * response
    caused = rng.poisson(expected)
    baseline_expected = config.baseline_daily_conversions * demand
    baseline = rng.poisson(baseline_expected)

    # Purchases: cause 0..nc-1 is a paid channel, nc the baseline, nc + 1 a repeat.
    first_day = np.concatenate(
        [np.repeat(np.arange(n), caused[i]) for i in range(nc)]
        + [np.repeat(np.arange(n), baseline)]
    )
    first_cause = np.concatenate(
        [np.full(caused[i].sum(), i) for i in range(nc)] + [np.full(baseline.sum(), nc)]
    )
    customer = np.arange(len(first_day))
    days_list, causes, customers = [first_day], [first_cause], [customer]
    gen_day, gen_customer = first_day, customer
    while len(gen_day):
        again = rng.random(len(gen_day)) < config.repeat_probability
        gap = repeat_gaps(rng, config.repeat_gap_days, int(again.sum()))
        day = gen_day[again] + gap
        keep = day < n
        gen_day, gen_customer = day[keep], gen_customer[again][keep]
        days_list.append(gen_day)
        causes.append(np.full(len(gen_day), nc + 1))
        customers.append(gen_customer)
    p_day = np.concatenate(days_list)
    p_cause = np.concatenate(causes)
    p_customer = np.concatenate(customers)
    order = np.lexsort((rng.random(len(p_day)), p_day))
    p_day, p_cause, p_customer = p_day[order], p_cause[order], p_customer[order]
    p_value = _order_values_cents(rng, config, len(p_day))

    # Session keys: every paid campaign, then the unpaid sources.
    keys: list[tuple[str, str, str, str]] = []
    key_index: dict[tuple[str, str], int] = {}
    for ch in chans:
        for camp in ch.campaigns:
            key_index[(ch.name, camp)] = len(keys)
            keys.append((ch.source, ch.medium, camp, ch.channel_group))
    for name, (source, medium, campaign, group, _) in UNPAID.items():
        key_index[(name, "")] = len(keys)
        keys.append((source, medium, campaign, group))
    nk = len(keys)
    direct = key_index[("direct", "")]

    counts = np.zeros((n, nk), dtype=np.int64)
    for i, ch in enumerate(chans):
        landed = rng.binomial(clicks[i], config.landing_rate)
        split = rng.multinomial(landed, ch.campaign_shares)
        for j, camp in enumerate(ch.campaigns):
            counts[:, key_index[(ch.name, camp)]] = split[:, j]
    for name, (*_, mean_sessions) in UNPAID.items():
        counts[:, key_index[(name, "")]] = rng.poisson(mean_sessions * demand)

    # The last click before each purchase.
    def unpaid_or_brand(names: np.ndarray) -> np.ndarray:
        lookup = {k: key_index[(k, "")] for k in UNPAID} | {
            "brand_search": key_index[("paid_search", "brand")]
        }
        return np.array([lookup[x] for x in names], dtype=np.int64)

    dest = np.empty(len(p_day), dtype=np.int64)
    for i, ch in enumerate(chans):
        mine = np.flatnonzero(p_cause == i)
        own = rng.random(len(mine)) < ch.last_click_share
        camp = rng.choice(len(ch.campaigns), size=own.sum(), p=ch.campaign_shares)
        dest[mine[own]] = [key_index[(ch.name, ch.campaigns[c])] for c in camp]
        dest[mine[~own]] = unpaid_or_brand(_draw(rng, stolen_from(ch), (~own).sum()))
    for cause, rule in ((nc, BASELINE_LAST_CLICK), (nc + 1, REPEAT_LAST_CLICK)):
        mine = np.flatnonzero(p_cause == cause)
        dest[mine] = unpaid_or_brand(_draw(rng, rule, len(mine)))

    def ranks(groups: np.ndarray) -> np.ndarray:
        """Each element's position among the elements of its group, in array order."""
        order = np.argsort(groups, kind="stable")
        sorted_groups = groups[order]
        starts = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
        lengths = np.diff(np.r_[starts, len(groups)])
        rank_sorted = np.arange(len(groups)) - np.repeat(starts, lengths)
        out = np.empty(len(groups), dtype=np.int64)
        out[order] = rank_sorted
        return out

    group = p_day * nk + dest
    over = ranks(group) >= counts.ravel()[group]
    fallback = int((over & (dest != direct)).sum())
    dest[over & (dest != direct)] = direct
    group = p_day * nk + dest
    rank = ranks(group)
    needed = np.zeros(n * nk, dtype=np.int64)
    np.maximum.at(needed, group, rank + 1)
    flat = counts.ravel()
    added = int(np.maximum(needed - flat, 0).sum())
    flat[:] = np.maximum(flat, needed)
    starts = np.r_[0, np.cumsum(flat)[:-1]]

    # Sessions, in (day, key) order; purchase p takes the rank-th session of its group.
    s_day = np.repeat(np.repeat(np.arange(n), nk), flat)
    s_key = np.repeat(np.tile(np.arange(nk), n), flat)
    total = len(s_day)
    s_purchase = np.full(total, -1, dtype=np.int64)
    s_purchase[starts[group] + rank] = np.arange(len(p_day))
    has = s_purchase >= 0

    # Users: a customer keeps one id across purchases; other sessions come from a pool of
    # visitors, so some visit more than once.
    n_customers = int(p_customer.max()) + 1 if len(p_customer) else 0
    pool = max(1, int(0.7 * (total - has.sum())))
    s_user = rng.integers(n_customers, n_customers + pool, size=total)
    s_user[has] = p_customer[s_purchase[has]]

    # Visits before each purchase: some of the visitor sessions in the days before it
    # become the customer's, so the path to a purchase is in the table. Each is taken
    # from that day's sessions without a purchase, in random order; a day with too few
    # (never seen at the default sizes) leaves the rest out.
    n_visits = rng.poisson(config.visits_before_purchase, len(p_day))
    v_purchase = np.repeat(np.arange(len(p_day)), n_visits)
    v_day = p_day[v_purchase] - rng.integers(1, config.visit_window_days + 1, len(v_purchase))
    v_purchase, v_day = v_purchase[v_day >= 0], v_day[v_day >= 0]
    free = np.flatnonzero(~has)
    free = free[np.lexsort((rng.random(len(free)), s_day[free]))]
    free_start = np.searchsorted(s_day[free], np.arange(n))
    free_count = np.bincount(s_day[free], minlength=n)
    v_rank = ranks(v_day)
    fits = v_rank < free_count[v_day]
    s_user[free[free_start[v_day[fits]] + v_rank[fits]]] = p_customer[v_purchase[fits]]

    s_seconds = rng.integers(0, 86400, size=total)
    s_device = rng.choice(len(DEVICES), size=total, p=DEVICE_SHARES)
    s_engaged = has | (rng.random(total) < config.engaged_rate)
    # Distinct per user, so no two users can share a pseudo id.
    user_number = 10**9 + rng.choice(2**31 - 1 - 10**9, size=n_customers + pool, replace=False)
    start_epoch = (np.datetime64(config.start, "s") - np.datetime64(0, "s")).astype(np.int64)
    s_epoch = start_epoch + s_day.astype(np.int64) * 86400 + s_seconds
    first_seen = np.full(n_customers + pool, np.iinfo(np.int64).max)
    np.minimum.at(first_seen, s_user, s_epoch)

    by_time = np.lexsort((s_key, s_epoch))
    s = {
        "day": s_day[by_time],
        "epoch": s_epoch[by_time],
        "user": s_user[by_time],
        "user_number": user_number[s_user[by_time]],
        "first_seen": first_seen[s_user[by_time]],
        "key": s_key[by_time],
        "device": s_device[by_time],
        "engaged": s_engaged[by_time],
        "purchase": s_purchase[by_time],
    }
    # What caused each purchase (-1 for a session without one), for checking attribution;
    # not written to the session table, which shows only what GA4 would.
    s["cause"] = np.where(s["purchase"] >= 0, p_cause[s["purchase"]], -1)
    # Transaction ids follow time order.
    p_transaction = np.full(len(p_day), -1, dtype=np.int64)
    bought = s["purchase"][s["purchase"] >= 0]
    p_transaction[bought] = np.arange(1, len(bought) + 1)
    s["transaction"] = np.where(s["purchase"] >= 0, p_transaction[s["purchase"]], -1)
    s["revenue_cents"] = np.where(s["purchase"] >= 0, p_value[s["purchase"]], 0)

    # Caused purchases and revenue per channel and day.
    conv = np.zeros((nc, n), dtype=np.int64)
    rev = np.zeros((nc, n), dtype=np.int64)
    paid = p_cause < nc
    np.add.at(conv, (p_cause[paid], p_day[paid]), 1)
    np.add.at(rev, (p_cause[paid], p_day[paid]), p_value[paid])
    channel_daily = {
        "day": np.tile(np.arange(n), nc),
        "channel": np.repeat(np.arange(nc), n),
        "spend_cents": spend.ravel(),
        "impressions": impressions.ravel(),
        "clicks": clicks.ravel(),
        "conversions": conv.ravel(),
        "revenue_cents": rev.ravel(),
    }
    order = np.lexsort((channel_daily["channel"], channel_daily["day"]))
    channel_daily = {k: v[order] for k, v in channel_daily.items()}

    # Truth per day and component: each channel, the baseline, and repeats.
    comp_conv = np.zeros((nc + 2, n), dtype=np.int64)
    comp_rev = np.zeros((nc + 2, n), dtype=np.int64)
    np.add.at(comp_conv, (p_cause, p_day), 1)
    np.add.at(comp_rev, (p_cause, p_day), p_value)
    nan_row = np.full(n, np.nan)
    truth_daily = {
        "day": np.tile(np.arange(n), nc + 2),
        "component": np.repeat(np.arange(nc + 2), n),
        "adstocked_spend": np.concatenate([*adstocked, nan_row, nan_row]),
        "expected_conversions": np.concatenate([*expected, baseline_expected, nan_row]),
        "conversions": comp_conv.ravel(),
        "revenue_cents": comp_rev.ravel(),
    }
    order = np.lexsort((truth_daily["component"], truth_daily["day"]))
    truth_daily = {k: v[order] for k, v in truth_daily.items()}

    return Dataset(
        config=config,
        dates=np.array(dates, dtype="datetime64[D]"),
        demand=demand,
        channel_daily=channel_daily,
        truth_daily=truth_daily,
        sessions=s,
        session_keys=keys,
        betas=betas,
        fallback_purchases=fallback,
        added_direct_sessions=added,
    )


def _money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(int(cents))
    return f"{sign}{cents // 100}.{cents % 100:02d}"


def component_names(config: GeneratorConfig) -> list[str]:
    return [c.name for c in config.channels] + ["baseline", "repeat"]


def realised_rates(data: Dataset) -> dict[str, dict[str, float | None]]:
    """Each channel's rates over the whole run, from the generated rows."""
    cd = data.channel_daily
    out = {}
    for i, ch in enumerate(data.config.channels):
        m = cd["channel"] == i
        spend = cd["spend_cents"][m].sum() / 100
        clicks = int(cd["clicks"][m].sum())
        impressions = int(cd["impressions"][m].sum())
        conversions = int(cd["conversions"][m].sum())
        revenue = cd["revenue_cents"][m].sum() / 100
        out[ch.name] = {
            "spend": round(spend, 2),
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "revenue": round(revenue, 2),
            # None (null in truth.json) where the denominator is zero: JSON has no NaN.
            "ctr": clicks / impressions if impressions else None,
            "cpc": spend / clicks if clicks else None,
            "cvr": conversions / clicks if clicks else None,
            "cac": spend / conversions if conversions else None,
            "roas": revenue / spend if spend else None,
            "zero_spend_days": int((cd["spend_cents"][m] == 0).sum()),
        }
    return out


def truth(data: Dataset) -> dict:
    """The run's configuration, true parameters and realised rates, for truth.json."""
    config = data.config
    td = data.truth_daily
    comps = component_names(config)
    totals = {
        name: {
            "conversions": int(td["conversions"][td["component"] == i].sum()),
            "revenue": round(td["revenue_cents"][td["component"] == i].sum() / 100, 2),
        }
        for i, name in enumerate(comps)
    }
    cfg = asdict(config)
    cfg.pop("channels")
    cfg["start"] = config.start.isoformat()
    return {
        "generator": "marketing_roi.generate",
        "config": cfg,
        "period": {"first": str(data.dates[0]), "last": str(data.dates[-1]), "days": config.days},
        "adstock": "normalised geometric: a[t] = (1 - decay) * spend[t] + decay * a[t-1]",
        "response": "expected purchases = beta * a^shape / (a^shape + half_saturation^shape)",
        "channels": {
            ch.name: {
                "source": ch.source,
                "medium": ch.medium,
                "channel_group": ch.channel_group,
                "daily_spend_target": ch.daily_spend,
                "cpc": ch.cpc,
                "ctr": ch.ctr,
                "cvr": ch.cvr,
                "implied_cac": round(ch.cac, 2),
                "decay": ch.decay,
                "half_saturation_dollars": ch.half_saturation * ch.daily_spend,
                "hill_shape": ch.hill_shape,
                "beta": data.betas[ch.name],
                "last_click_share": ch.last_click_share,
                "flighted": ch.flighted,
                "rate_sources": ch.rate_sources,
            }
            for ch in config.channels
        },
        "last_click_rules": {
            "paid_channel_not_own": {ch.name: stolen_from(ch) for ch in config.channels},
            "baseline": BASELINE_LAST_CLICK,
            "repeat": REPEAT_LAST_CLICK,
        },
        "unpaid_sessions_per_day": {k: v[4] for k, v in UNPAID.items()},
        "realised": realised_rates(data),
        "purchases_by_component": totals,
        "rows": {
            "channel_daily": len(data.channel_daily["day"]),
            "ga4_sessions": len(data.sessions["day"]),
            "truth_daily": len(data.truth_daily["day"]),
        },
        "fallback_purchases": data.fallback_purchases,
        "added_direct_sessions": data.added_direct_sessions,
    }


def write(data: Dataset, out_dir: Path) -> dict[str, Path]:
    """Write the three tables and truth.json into `out_dir`; return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day_str = [str(d) for d in data.dates]
    names = [c.name for c in data.config.channels]
    paths = {k: out_dir / f"{k}.csv" for k in ("channel_daily", "ga4_sessions", "truth_daily")}

    cd = data.channel_daily
    with paths["channel_daily"].open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CHANNEL_DAILY_COLUMNS)
        for d, c, sp, im, cl, cv, rv in zip(
            cd["day"],
            cd["channel"],
            cd["spend_cents"],
            cd["impressions"],
            cd["clicks"],
            cd["conversions"],
            cd["revenue_cents"],
            strict=True,
        ):
            w.writerow((day_str[d], names[c], _money(sp), int(im), int(cl), int(cv), _money(rv)))

    s = data.sessions
    keys = data.session_keys
    with paths["ga4_sessions"].open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SESSION_COLUMNS)
        for i in range(len(s["day"])):
            source, medium, campaign, group = keys[s["key"][i]]
            stamp = dt.datetime.fromtimestamp(int(s["epoch"][i]), dt.UTC).replace(tzinfo=None)
            bought = s["transaction"][i] >= 0
            w.writerow(
                (
                    i + 1,
                    stamp.isoformat(sep=" "),
                    f"{s['user_number'][i]}.{s['first_seen'][i]}",
                    source,
                    medium,
                    campaign,
                    group,
                    DEVICES[s["device"][i]],
                    "true" if s["engaged"][i] else "false",
                    f"T{s['transaction'][i]:08d}" if bought else "",
                    _money(s["revenue_cents"][i]) if bought else "",
                )
            )

    td = data.truth_daily
    comps = component_names(data.config)
    with paths["truth_daily"].open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TRUTH_DAILY_COLUMNS)
        for d, c, a, e, cv, rv in zip(
            td["day"],
            td["component"],
            td["adstocked_spend"],
            td["expected_conversions"],
            td["conversions"],
            td["revenue_cents"],
            strict=True,
        ):
            w.writerow(
                (
                    day_str[d],
                    comps[c],
                    "" if math.isnan(a) else f"{a:.4f}",
                    "" if math.isnan(e) else f"{e:.6f}",
                    int(cv),
                    _money(rv),
                )
            )

    paths["truth"] = out_dir / "truth.json"
    paths["truth"].write_text(
        json.dumps(truth(data), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return paths


def _show(value: float | None, fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m marketing_roi.generate",
        description="Generate the synthetic marketing data (see the module docstring).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/generated"),
        help="folder to write into (default: data/generated)",
    )
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument(
        "--start",
        type=dt.date.fromisoformat,
        default=GeneratorConfig.start,
        help="first day, YYYY-MM-DD (default: 2024-01-01)",
    )
    parser.add_argument("--days", type=int, default=GeneratorConfig.days)
    args = parser.parse_args(argv)
    try:
        config = GeneratorConfig(seed=args.seed, start=args.start, days=args.days)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.out.exists() and not args.out.is_dir():
        parser.error(f"--out {args.out} is a file, not a folder")
    data = generate(config)
    try:
        paths = write(data, args.out)
    except OSError as exc:
        print(f"error: cannot write to {args.out}: {exc}", file=sys.stderr)
        return 1
    summary = truth(data)
    print(f"wrote {args.out}: " + ", ".join(f"{k} {v:,} rows" for k, v in summary["rows"].items()))
    for name, r in summary["realised"].items():
        print(
            f"  {name:13s} spend ${r['spend']:>12,.2f}  CTR {_show(r['ctr'], '.4f')}"
            f"  CPC ${_show(r['cpc'], '.2f')}  CVR {_show(r['cvr'], '.4f')}"
            f"  CAC ${_show(r['cac'], '.2f')}  ROAS {_show(r['roas'], '.2f')}"
        )
    print(f"  truth: {paths['truth']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
