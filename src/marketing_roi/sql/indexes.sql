-- Secondary indexes, built once over the loaded rows rather than updated row by
-- row during the copy. Each comment names the query it serves.

-- Daily traffic by source: sessions for one or more days, grouped by source.
CREATE INDEX ga4_sessions_date_source ON marketing.ga4_sessions (session_date, source_id);

-- Purchases only (about 1.5% of sessions): revenue and last-click credit by
-- day, without reading the sessions that bought nothing.
CREATE INDEX ga4_purchases_by_date ON marketing.ga4_sessions (session_date, source_id)
    INCLUDE (purchase_revenue) WHERE transaction_id IS NOT NULL;

-- A customer's purchases in order: first purchase (acquisition cohort and
-- channel) and the repeats after it.
CREATE INDEX ga4_purchases_by_user ON marketing.ga4_sessions (user_pseudo_id, session_start)
    INCLUDE (source_id, purchase_revenue) WHERE transaction_id IS NOT NULL;

-- One channel's days in order (the primary key leads with the date).
CREATE INDEX channel_daily_by_channel ON marketing.channel_daily (channel_id, date);
