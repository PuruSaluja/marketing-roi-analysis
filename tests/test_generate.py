import csv
import dataclasses
import datetime as dt
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from marketing_roi.channels import CHANNELS, channel
from marketing_roi.generate import (
    CHANNEL_DAILY_COLUMNS,
    SESSION_COLUMNS,
    TRUTH_DAILY_COLUMNS,
    GeneratorConfig,
    adstock,
    black_friday,
    generate,
    hill,
    main,
    repeat_gaps,
    stolen_from,
    write,
)

SMALL = GeneratorConfig(seed=7, start=dt.date(2024, 10, 1), days=150)


@pytest.fixture(scope="module")
def small():
    return generate(SMALL)


@pytest.fixture(scope="module")
def full():
    return generate(GeneratorConfig())


@pytest.fixture(scope="module")
def written(small, tmp_path_factory):
    out = tmp_path_factory.mktemp("generated")
    paths = write(small, out)
    return small, paths


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- the transforms -------------------------------------------------------------------------


def test_adstock_carries_a_share_into_the_next_day() -> None:
    out = adstock(np.array([100.0, 0.0, 0.0, 50.0]), 0.5)
    # Starts as if day 0's spend had always run, then decays by half a day.
    assert out.tolist() == [100.0, 50.0, 25.0, 37.5]


def test_adstock_of_a_constant_spend_is_that_spend() -> None:
    assert np.allclose(adstock(np.full(30, 80.0), 0.7), 80.0)


def test_adstock_without_decay_is_the_spend() -> None:
    spend = np.array([3.0, 0.0, 9.0])
    assert adstock(spend, 0.0).tolist() == spend.tolist()


def test_hill_is_half_at_the_half_saturation_point() -> None:
    assert hill(np.array([0.0]), 10.0, 1.3)[0] == 0.0
    assert hill(np.array([10.0]), 10.0, 1.3)[0] == pytest.approx(0.5)
    assert hill(np.array([20.0]), 10.0, 1.0)[0] == pytest.approx(2 / 3)
    assert hill(np.array([1e9]), 10.0, 1.0)[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2023, dt.date(2023, 11, 24)),
        (2024, dt.date(2024, 11, 29)),
        (2025, dt.date(2025, 11, 28)),
        (2026, dt.date(2026, 11, 27)),
    ],
)
def test_black_friday_is_the_day_after_the_fourth_thursday(year, expected) -> None:
    assert black_friday(year) == expected


# --- configuration --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", "42"),
        ("seed", 1.5),
        ("seed", True),
        ("days", 10.0),
        ("aov", "90"),
        ("repeat_probability", None),
        ("start", "2024-01-01"),
        ("start", dt.datetime(2024, 1, 1)),
    ],
)
def test_a_setting_of_the_wrong_type_is_refused(field, value) -> None:
    with pytest.raises(TypeError, match=field):
        GeneratorConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", -1),
        ("days", 0),
        ("aov", 0.0),
        ("aov", float("nan")),
        ("landing_rate", 0.0),
        ("landing_rate", 1.01),
        ("repeat_probability", 0.99),
        ("repeat_gap_days", 0.5),
        ("seasonal_amplitude", float("inf")),
    ],
)
def test_a_setting_out_of_range_is_refused(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        GeneratorConfig(**{field: value})


def test_channels_must_include_branded_search() -> None:
    without_search = tuple(c for c in CHANNELS if c.name != "paid_search")
    with pytest.raises(ValueError, match="paid_search"):
        GeneratorConfig(channels=without_search)
    duplicated = (*CHANNELS, CHANNELS[0])
    with pytest.raises(ValueError, match="distinct"):
        GeneratorConfig(channels=duplicated)


# --- determinism ----------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_files(small, tmp_path) -> None:
    a = write(small, tmp_path / "a")
    b = write(generate(SMALL), tmp_path / "b")
    for key in a:
        assert a[key].read_bytes() == b[key].read_bytes(), key


def test_another_seed_gives_other_data(small) -> None:
    other = generate(dataclasses.replace(SMALL, seed=8))
    assert not np.array_equal(
        other.channel_daily["spend_cents"], small.channel_daily["spend_cents"]
    )


# --- the tables agree with each other -------------------------------------------------------


def test_purchases_in_sessions_equal_the_components_each_day(small) -> None:
    td, s = small.truth_daily, small.sessions
    n = small.config.days
    comp_conv = np.bincount(td["day"], weights=td["conversions"], minlength=n)
    comp_rev = np.bincount(td["day"], weights=td["revenue_cents"], minlength=n)
    bought = s["transaction"] >= 0
    sess_conv = np.bincount(s["day"][bought], minlength=n)
    sess_rev = np.bincount(s["day"][bought], weights=s["revenue_cents"][bought], minlength=n)
    assert np.array_equal(comp_conv, sess_conv)
    assert np.array_equal(comp_rev, sess_rev)


def test_channel_daily_is_the_truth_s_channel_rows(small) -> None:
    cd, td = small.channel_daily, small.truth_daily
    channel_rows = td["component"] < len(small.config.channels)
    assert np.array_equal(cd["conversions"], td["conversions"][channel_rows])
    assert np.array_equal(cd["revenue_cents"], td["revenue_cents"][channel_rows])


def test_paid_sessions_never_exceed_clicks(small) -> None:
    cd, s = small.channel_daily, small.sessions
    groups = {
        (src, med): i
        for i, c in enumerate(small.config.channels)
        for src, med in [(c.source, c.medium)]
    }
    sessions = Counter()
    for day, key in zip(s["day"], s["key"], strict=True):
        source, medium, _, _ = small.session_keys[key]
        if (source, medium) in groups:
            sessions[(day, groups[(source, medium)])] += 1
    for day, ch, clicks in zip(cd["day"], cd["channel"], cd["clicks"], strict=True):
        assert sessions[(day, ch)] <= clicks


def test_funnel_counts_are_ordered(small) -> None:
    cd = small.channel_daily
    assert (cd["spend_cents"] >= 0).all()
    assert (cd["clicks"] <= cd["impressions"]).all()
    assert (cd["clicks"][cd["spend_cents"] == 0] == 0).all()


def test_every_purchase_has_its_own_session_and_transaction(small) -> None:
    s = small.sessions
    ids = s["transaction"][s["transaction"] >= 0]
    assert len(ids) == len(set(ids.tolist()))
    assert sorted(ids.tolist()) == list(range(1, len(ids) + 1))
    assert (s["revenue_cents"][s["transaction"] >= 0] > 0).all()
    assert (s["revenue_cents"][s["transaction"] < 0] == 0).all()
    assert s["engaged"][s["transaction"] >= 0].all()


def test_sessions_are_in_time_order_and_a_user_first_seen_no_later(small) -> None:
    s = small.sessions
    assert (np.diff(s["epoch"]) >= 0).all()
    assert (s["first_seen"] <= s["epoch"]).all()
    first = {}
    for user, epoch in zip(s["user"], s["epoch"], strict=True):
        first.setdefault(user, epoch)
    assert all(s["first_seen"][i] == first[u] for i, u in enumerate(s["user"]))


def test_repeat_customers_keep_their_id(small) -> None:
    s = small.sessions
    bought = s["transaction"] >= 0
    per_user = Counter(s["user"][bought].tolist())
    assert max(per_user.values()) > 1


def test_flighted_channels_go_dark_and_carry_over(full) -> None:
    cd, td = full.channel_daily, full.truth_daily
    for i, ch in enumerate(full.config.channels):
        dark = (cd["channel"] == i) & (cd["spend_cents"] == 0)
        if not ch.flighted:
            assert not dark.any(), ch.name
            continue
        assert dark.sum() > 30, ch.name
        # Adstock still holds spend from before the flight ended, so purchases continue.
        rows = td["component"] == i
        expected_on_dark_days = td["expected_conversions"][rows][dark[cd["channel"] == i]]
        assert (expected_on_dark_days > 0).any(), ch.name
        assert cd["conversions"][dark].sum() > 0, ch.name


# --- calibration and attribution over the full default run ----------------------------------


def test_the_default_run_lands_on_the_benchmark_rates(full) -> None:
    cd = full.channel_daily
    for i, ch in enumerate(full.config.channels):
        m = cd["channel"] == i
        spend = cd["spend_cents"][m].sum() / 100
        clicks = cd["clicks"][m].sum()
        assert spend / full.config.days == pytest.approx(ch.daily_spend, rel=1e-4), ch.name
        assert clicks / cd["impressions"][m].sum() == pytest.approx(ch.ctr, rel=0.02), ch.name
        assert spend / clicks == pytest.approx(ch.cpc, rel=0.02), ch.name
        # Purchases are Poisson around beta * hill(adstock); the beta is set from expected
        # clicks, so realised CVR sits within a few percent of the target.
        assert cd["conversions"][m].sum() / clicks == pytest.approx(ch.cvr, rel=0.06), ch.name


def test_last_click_overcredits_search_and_undercredits_video(full) -> None:
    s, cd = full.sessions, full.channel_daily
    bought = s["transaction"] >= 0
    groups = Counter(full.session_keys[k][3] for k in s["key"][bought])
    caused = {
        c.channel_group: cd["conversions"][cd["channel"] == i].sum()
        for i, c in enumerate(full.config.channels)
    }
    assert groups["Paid Search"] > caused["Paid Search"]
    assert groups["Display"] < caused["Display"] / 3
    assert groups["Paid Video"] < caused["Paid Video"] / 3


def test_hardly_any_purchase_needs_a_fallback_session(full) -> None:
    purchases = (full.sessions["transaction"] >= 0).sum()
    assert full.fallback_purchases < 0.002 * purchases
    assert full.added_direct_sessions == 0


def test_holiday_demand_is_higher(full) -> None:
    dates = full.dates.astype(object)
    bf = [i for i, d in enumerate(dates) if d == black_friday(d.year)]
    assert len(bf) == 2
    for i in bf:
        assert full.demand[i] > 1.4 * full.demand[i - 7]


# --- the written files ----------------------------------------------------------------------


def test_files_have_the_documented_columns(written) -> None:
    _, paths = written
    for key, columns in (
        ("channel_daily", CHANNEL_DAILY_COLUMNS),
        ("ga4_sessions", SESSION_COLUMNS),
        ("truth_daily", TRUTH_DAILY_COLUMNS),
    ):
        with paths[key].open(newline="", encoding="utf-8") as f:
            assert tuple(next(csv.reader(f))) == columns


def test_written_money_adds_up_to_the_cent(written) -> None:
    data, paths = written
    sessions = read_csv(paths["ga4_sessions"])
    truth_rows = read_csv(paths["truth_daily"])
    by_day = defaultdict(Decimal)
    for r in sessions:
        if r["transaction_id"]:
            by_day[r["session_start"][:10]] += Decimal(r["purchase_revenue"])
    truth_by_day = defaultdict(Decimal)
    for r in truth_rows:
        truth_by_day[r["date"]] += Decimal(r["revenue"])
    assert by_day == truth_by_day
    assert len(sessions) == len(data.sessions["day"])


def test_ga4_ids_have_the_export_s_shape(written) -> None:
    _, paths = written
    rows = read_csv(paths["ga4_sessions"])
    assert [int(r["session_id"]) for r in rows] == list(range(1, len(rows) + 1))
    for r in rows[:500]:
        number, first_seen = r["user_pseudo_id"].split(".")
        assert number.isdigit() and first_seen.isdigit()
        start = dt.datetime.fromisoformat(r["session_start"]).replace(tzinfo=dt.UTC)
        assert int(first_seen) <= start.timestamp()
    tx = [r["transaction_id"] for r in rows if r["transaction_id"]]
    assert all(t.startswith("T") and len(t) == 9 for t in tx)


def test_truth_json_records_parameters_and_realised_rates(written) -> None:
    data, paths = written
    t = json.loads(paths["truth"].read_text(encoding="utf-8"))
    assert t["config"]["seed"] == SMALL.seed
    assert t["period"] == {"first": "2024-10-01", "last": "2025-02-27", "days": 150}
    search = t["channels"]["paid_search"]
    assert search["half_saturation_dollars"] == pytest.approx(
        channel("paid_search").half_saturation * channel("paid_search").daily_spend
    )
    assert "wordstream.com" in search["rate_sources"]["cpc"]
    assert (search["source"], search["medium"], search["channel_group"]) == (
        "google",
        "cpc",
        "Paid Search",
    )
    assert t["channels"]["display"]["rate_sources"]["cvr"].startswith("assumed")
    total = sum(v["conversions"] for v in t["purchases_by_component"].values())
    assert total == int((data.sessions["transaction"] >= 0).sum())
    assert t["rows"]["ga4_sessions"] == len(data.sessions["day"])


def test_cli_writes_the_files(tmp_path, capsys) -> None:
    assert main(["--out", str(tmp_path), "--seed", "3", "--days", "40"]) == 0
    assert {p.name for p in tmp_path.iterdir()} == {
        "channel_daily.csv",
        "ga4_sessions.csv",
        "truth_daily.csv",
        "truth.json",
    }
    assert "paid_search" in capsys.readouterr().out


def test_cli_refuses_a_bad_setting(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--out", str(tmp_path), "--days", "0"])
    assert exc.value.code == 2
    assert "days must be in" in capsys.readouterr().err


# --- what the generator promises ------------------------------------------------------------


def test_repeat_gaps_average_the_setting() -> None:
    gaps = repeat_gaps(np.random.default_rng(0), 75.0, 1_000_000)
    assert gaps.min() >= 1
    assert np.issubdtype(gaps.dtype, np.integer)
    assert gaps.mean() == pytest.approx(75.0, abs=0.3)


def test_each_channel_gets_its_last_click_share_on_days_it_has_sessions(full) -> None:
    s, n = full.sessions, full.config.days
    groups = np.array([k[3] for k in full.session_keys])
    for i, ch in enumerate(full.config.channels):
        is_own = groups[s["key"]] == ch.channel_group
        own_sessions = np.bincount(s["day"], weights=is_own, minlength=n)
        caused = s["cause"] == i
        on = caused & (own_sessions[s["day"]] > 0)
        share = is_own[on].mean()
        sd = (ch.last_click_share * (1 - ch.last_click_share) / on.sum()) ** 0.5
        assert abs(share - ch.last_click_share) < 4 * sd, (ch.name, share)


def test_paid_search_never_loses_a_purchase_to_its_own_brand_campaign() -> None:
    options = stolen_from(channel("paid_search"))
    assert "brand_search" not in options
    assert sum(options.values()) == pytest.approx(1.0)
    assert "brand_search" in stolen_from(channel("display"))


def test_every_user_has_a_distinct_number(full) -> None:
    users, first = np.unique(full.sessions["user"], return_index=True)
    assert len(np.unique(full.sessions["user_number"][first])) == len(users)


def test_truth_json_is_strict_json_when_a_channel_has_no_clicks(tmp_path) -> None:
    starved = dataclasses.replace(channel("display"), daily_spend=0.01)
    channels = tuple(starved if c.name == "display" else c for c in CHANNELS)
    paths = write(generate(GeneratorConfig(days=5, channels=channels)), tmp_path)

    def refuse(constant: str) -> None:
        raise ValueError(f"not JSON: {constant}")

    t = json.loads(paths["truth"].read_text(encoding="utf-8"), parse_constant=refuse)
    display = t["realised"]["display"]
    assert display["clicks"] == 0
    assert display["cpc"] is None and display["cvr"] is None and display["cac"] is None


def test_cli_refuses_an_output_path_that_is_a_file(tmp_path, capsys) -> None:
    target = tmp_path / "taken"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["--out", str(target), "--days", "3"])
    assert exc.value.code == 2
    assert "is a file, not a folder" in capsys.readouterr().err
    assert target.read_text(encoding="utf-8") == "x"
