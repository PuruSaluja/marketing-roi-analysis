-- Tables for the generated data.
--
-- `marketing` holds what a business would see: spend and funnel per channel and
-- day, and GA4-style sessions. `truth` holds what only the generator knows (the
-- purchases each component caused, and its parameters), kept apart so analysis
-- queries cannot use it by accident.

CREATE SCHEMA marketing;
CREATE SCHEMA truth;

CREATE TABLE marketing.channels (
    channel_id    smallint PRIMARY KEY,
    name          text NOT NULL UNIQUE,
    source        text NOT NULL,
    medium        text NOT NULL,
    channel_group text NOT NULL,
    flighted      boolean NOT NULL
);

-- Spend and funnel per day and paid channel. conversions and revenue are the
-- purchases the channel's spend caused, carryover included.
--
-- numeric and double precision accept NaN, and PostgreSQL sorts NaN above every
-- number, so "x >= 0" alone lets it in; money columns also say "x <> 'NaN'", and
-- the truth table's floats "x < 'Infinity'", which NaN fails as well.
CREATE TABLE marketing.channel_daily (
    date        date NOT NULL,
    channel_id  smallint NOT NULL REFERENCES marketing.channels,
    spend       numeric(12, 2) NOT NULL CHECK (spend >= 0 AND spend <> 'NaN'),
    impressions bigint NOT NULL CHECK (impressions >= 0),
    clicks      bigint NOT NULL CHECK (clicks >= 0),
    conversions integer NOT NULL CHECK (conversions >= 0),
    revenue     numeric(12, 2) NOT NULL CHECK (revenue >= 0 AND revenue <> 'NaN'),
    PRIMARY KEY (date, channel_id),
    CHECK (clicks <= impressions)
);

-- One row per source / medium / campaign, so sessions carry a small key.
CREATE TABLE marketing.traffic_sources (
    source_id             smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source                text NOT NULL,
    medium                text NOT NULL,
    campaign              text NOT NULL,
    default_channel_group text NOT NULL,
    channel_id            smallint REFERENCES marketing.channels,
    UNIQUE (source, medium, campaign)
);

-- GA4-style sessions. A purchase is credited to the session it happened in,
-- which is the last click before it.
CREATE TABLE marketing.ga4_sessions (
    session_id       bigint PRIMARY KEY,
    session_start    timestamp NOT NULL,
    session_date     date GENERATED ALWAYS AS (session_start::date) STORED,
    user_pseudo_id   text NOT NULL,
    source_id        smallint NOT NULL REFERENCES marketing.traffic_sources,
    device_category  text NOT NULL CHECK (device_category IN ('mobile', 'desktop', 'tablet')),
    engaged_session  boolean NOT NULL,
    transaction_id   text UNIQUE,
    purchase_revenue numeric(10, 2) CHECK (purchase_revenue > 0 AND purchase_revenue <> 'NaN'),
    CHECK ((transaction_id IS NULL) = (purchase_revenue IS NULL)),
    CHECK (transaction_id IS NULL OR engaged_session)
);

CREATE VIEW marketing.ga4_sessions_flat AS
SELECT s.session_id, s.session_start, s.session_date, s.user_pseudo_id,
       t.source, t.medium, t.campaign, t.default_channel_group,
       s.device_category, s.engaged_session, s.transaction_id, s.purchase_revenue
FROM marketing.ga4_sessions s
JOIN marketing.traffic_sources t USING (source_id);

-- What caused each day's purchases: each paid channel, the baseline, repeats.
CREATE TABLE truth.daily (
    date                 date NOT NULL,
    component            text NOT NULL,
    adstocked_spend      double precision
        CHECK (adstocked_spend >= 0 AND adstocked_spend < 'Infinity'),
    expected_conversions double precision
        CHECK (expected_conversions >= 0 AND expected_conversions < 'Infinity'),
    conversions          integer NOT NULL CHECK (conversions >= 0),
    revenue              numeric(12, 2) NOT NULL CHECK (revenue >= 0 AND revenue <> 'NaN'),
    PRIMARY KEY (date, component)
);

-- The generator's settings, true parameters and realised rates (truth.json).
CREATE TABLE truth.run (
    only_row  boolean PRIMARY KEY DEFAULT true CHECK (only_row),
    loaded_at timestamptz NOT NULL DEFAULT now(),
    params    jsonb NOT NULL
);
