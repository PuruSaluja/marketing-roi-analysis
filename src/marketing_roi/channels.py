"""The paid channels the generator simulates, and where their rates come from.

Cost per click, click-through rate and conversion rate are taken from published
benchmarks where one exists for an online shop, and marked as assumed where
none was found. Everything else here (adstock decay, saturation, how often a
channel gets the last click) is the generator's own choice: it is the hidden
truth a fitted model is later checked against, not a claim about any market.
"""

from __future__ import annotations

from dataclasses import dataclass

GOOGLE_ADS_2025 = (
    "WordStream, Google Ads Benchmarks 2025 (16,000+ US campaigns, April 2024 to "
    "March 2025), Shopping, Collectibles & Gifts: CTR 8.92%, CPC $3.49, "
    "CVR 3.83% (CPL $47.94, all industries $70.11). "
    "https://www.wordstream.com/blog/2025-google-ads-benchmarks"
)
FACEBOOK_ADS_2025 = (
    "WordStream, Facebook Ads Benchmarks 2025 (US campaigns, April 2024 to "
    "June 2025), traffic campaigns, Shopping, Collectibles & Gifts: CTR 4.13%, "
    "CPC $0.34. https://www.wordstream.com/blog/facebook-ads-benchmarks-2025"
)
ASSUMED = "assumed; no published benchmark for an online shop was used"


@dataclass(frozen=True)
class Channel:
    """One paid channel: its reporting names, its rates and its true response."""

    name: str
    source: str
    medium: str
    channel_group: str
    campaigns: tuple[str, ...]
    campaign_shares: tuple[float, ...]
    daily_spend: float
    """Mean daily spend over the whole period, in dollars."""
    cpc: float
    ctr: float
    cvr: float
    """Incremental purchases per click, on average over the period."""
    decay: float
    """Geometric adstock: the share of a day's adstocked spend carried into the next."""
    half_saturation: float
    """Adstocked daily spend, as a multiple of `daily_spend`, at which the response is half its
    ceiling."""
    hill_shape: float
    last_click_share: float
    """Share of the purchases this channel causes whose last click is one of its own sessions,
    on days it has sessions. A purchase it causes while a flight is off (carryover) has none
    to land on, and is credited to direct."""
    flighted: bool
    """Run in flights (weeks on, weeks off) rather than always on."""
    rate_sources: dict[str, str]

    @property
    def cac(self) -> float:
        """Spend per incremental purchase implied by the rates: cost per click over conversion
        rate."""
        return self.cpc / self.cvr


CHANNELS: tuple[Channel, ...] = (
    Channel(
        name="paid_search",
        source="google",
        medium="cpc",
        channel_group="Paid Search",
        campaigns=("brand", "generic", "competitor"),
        campaign_shares=(0.25, 0.60, 0.15),
        daily_spend=900.0,
        cpc=3.49,
        ctr=0.0892,
        cvr=0.0383,
        decay=0.10,
        half_saturation=0.8,
        hill_shape=1.2,
        last_click_share=0.85,
        flighted=False,
        rate_sources={"cpc": GOOGLE_ADS_2025, "ctr": GOOGLE_ADS_2025, "cvr": GOOGLE_ADS_2025},
    ),
    Channel(
        name="paid_social",
        source="facebook",
        medium="paid_social",
        channel_group="Paid Social",
        campaigns=("prospecting", "retargeting"),
        campaign_shares=(0.70, 0.30),
        daily_spend=600.0,
        cpc=0.34,
        ctr=0.0413,
        cvr=0.005,
        decay=0.40,
        half_saturation=1.2,
        hill_shape=1.5,
        last_click_share=0.45,
        flighted=False,
        rate_sources={"cpc": FACEBOOK_ADS_2025, "ctr": FACEBOOK_ADS_2025, "cvr": ASSUMED},
    ),
    Channel(
        name="display",
        source="google",
        medium="display",
        channel_group="Display",
        campaigns=("remarketing", "prospecting"),
        campaign_shares=(0.40, 0.60),
        daily_spend=300.0,
        cpc=0.60,
        ctr=0.005,
        cvr=0.004,
        decay=0.50,
        half_saturation=1.0,
        hill_shape=1.3,
        last_click_share=0.15,
        flighted=True,
        rate_sources={"cpc": ASSUMED, "ctr": ASSUMED, "cvr": ASSUMED},
    ),
    Channel(
        name="online_video",
        source="youtube",
        medium="cpv",
        channel_group="Paid Video",
        campaigns=("awareness",),
        campaign_shares=(1.0,),
        daily_spend=400.0,
        cpc=1.00,
        ctr=0.007,
        cvr=0.006,
        decay=0.70,
        half_saturation=2.0,
        hill_shape=1.0,
        last_click_share=0.10,
        flighted=True,
        rate_sources={"cpc": ASSUMED, "ctr": ASSUMED, "cvr": ASSUMED},
    ),
)


def channel(name: str) -> Channel:
    for c in CHANNELS:
        if c.name == name:
            return c
    raise KeyError(f"no channel named {name!r}")
