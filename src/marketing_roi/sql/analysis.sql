-- Analysis views: channel performance, customer acquisition cost, blended return on
-- spend, and customer cohorts.
--
-- They read only what a business would have: spend, impressions and clicks as the ad
-- platforms report them (marketing.channel_daily), and purchases as GA4 records them,
-- each credited to the last click before it (marketing.ga4_sessions). They never read
-- the truth schema, nor channel_daily's conversions and revenue, which are the
-- purchases each channel's spend caused and which only the generator knows. So every
-- figure here is a last-click figure, with the biases last click has.
--
-- Ratios are taken over sums (total spend over total clicks, not an average of daily
-- cost per click) and are NULL where the denominator is zero, as on days a flighted
-- channel is dark. A row whose month is NULL covers the whole period.

CREATE SCHEMA analysis;

-- The data's last day. Spend is reported for every day of the period, sessions only
-- for days that had some, so the period is taken from channel_daily.
CREATE VIEW analysis.period AS
SELECT min(date) AS first_day, max(date) AS last_day FROM marketing.channel_daily;

-- One row per day and paid channel: the platform's spend and funnel, and the sessions,
-- purchases and revenue GA4 credits to the channel's campaigns.
CREATE VIEW analysis.channel_daily AS
WITH ga4 AS (
    SELECT s.session_date AS date, t.channel_id,
           count(*) AS sessions,
           count(s.transaction_id) AS purchases,
           coalesce(sum(s.purchase_revenue), 0) AS revenue
    FROM marketing.ga4_sessions s
    JOIN marketing.traffic_sources t USING (source_id)
    WHERE t.channel_id IS NOT NULL
    GROUP BY 1, 2)
SELECT d.date, c.name AS channel, c.channel_group,
       d.spend, d.impressions, d.clicks,
       coalesce(g.sessions, 0) AS sessions,
       coalesce(g.purchases, 0) AS purchases,
       coalesce(g.revenue, 0) AS revenue
FROM marketing.channel_daily d
JOIN marketing.channels c USING (channel_id)
LEFT JOIN ga4 g USING (date, channel_id);

-- Channel performance by month, and over the whole period (month NULL).
CREATE VIEW analysis.channel_performance AS
SELECT date_trunc('month', date)::date AS month, channel, channel_group,
       sum(spend) AS spend,
       sum(impressions) AS impressions,
       sum(clicks) AS clicks,
       sum(sessions) AS sessions,
       sum(purchases) AS purchases,
       sum(revenue) AS revenue,
       sum(clicks)::numeric / nullif(sum(impressions), 0) AS ctr,
       sum(spend) / nullif(sum(clicks), 0) AS cpc,
       sum(purchases)::numeric / nullif(sum(clicks), 0) AS conversion_rate,
       sum(spend) / nullif(sum(purchases), 0) AS cost_per_purchase,
       sum(revenue) / nullif(sum(spend), 0) AS roas
FROM analysis.channel_daily
GROUP BY GROUPING SETS ((date_trunc('month', date), channel, channel_group),
                        (channel, channel_group));

-- Every purchase with its customer, and whether it was the customer's first in the
-- data. A customer is a user_pseudo_id. The generator keeps one per customer; real
-- GA4 sets one per browser or device, so there a customer who buys on two devices
-- counts as two, and one who first bought before the data starts counts as new.
CREATE VIEW analysis.purchases AS
SELECT s.session_id, s.session_start, s.session_date, s.user_pseudo_id,
       s.purchase_revenue AS revenue,
       t.default_channel_group AS channel_group,
       t.channel_id IS NOT NULL AS paid,
       row_number() OVER (PARTITION BY s.user_pseudo_id
                          ORDER BY s.session_start, s.session_id) = 1 AS first_purchase
FROM marketing.ga4_sessions s
JOIN marketing.traffic_sources t USING (source_id)
WHERE s.transaction_id IS NOT NULL;

-- One row per customer: when they first bought, and through which channel group.
CREATE VIEW analysis.customers AS
SELECT user_pseudo_id,
       session_start AS acquired_at,
       session_date AS acquired_on,
       date_trunc('month', session_date)::date AS cohort_month,
       channel_group AS acquired_via,
       paid AS acquired_paid,
       revenue AS first_order_revenue
FROM analysis.purchases
WHERE first_purchase;

-- Customer acquisition cost by month and channel group, and over the whole period.
-- A paid group's CAC is its spend over the customers whose first purchase it was
-- credited with by last click; unpaid groups have no spend and no CAC.
CREATE VIEW analysis.acquisition AS
WITH spend AS (
    SELECT date_trunc('month', date)::date AS month, channel_group, sum(spend) AS spend
    FROM analysis.channel_daily GROUP BY 1, 2),
acquired AS (
    SELECT cohort_month AS month, acquired_via AS channel_group, bool_or(acquired_paid) AS paid,
           count(*) AS new_customers
    FROM analysis.customers GROUP BY 1, 2),
monthly AS (
    SELECT month, channel_group,
           coalesce(a.paid, true) AS paid,  -- only paid groups have spend
           coalesce(s.spend, 0) AS spend,
           coalesce(a.new_customers, 0) AS new_customers
    FROM spend s FULL JOIN acquired a USING (month, channel_group))
SELECT month, channel_group, bool_or(paid) AS paid,
       sum(spend) AS spend,
       sum(new_customers) AS new_customers,
       CASE WHEN bool_or(paid) THEN sum(spend) / nullif(sum(new_customers), 0) END AS cac
FROM monthly
GROUP BY GROUPING SETS ((month, channel_group), (channel_group));

-- Blended figures by month and over the whole period: all paid spend against all
-- revenue and all new customers, whatever channel they came through. ROI is on
-- revenue, (revenue - spend) / spend: the data has no cost of goods, so it is not a
-- profit ROI.
CREATE VIEW analysis.blended AS
WITH spend AS (
    SELECT date_trunc('month', date)::date AS month, sum(spend) AS spend
    FROM marketing.channel_daily GROUP BY 1),
sales AS (
    SELECT date_trunc('month', session_date)::date AS month,
           count(*) AS purchases,
           sum(revenue) AS revenue,
           count(*) FILTER (WHERE first_purchase) AS new_customers,
           coalesce(sum(revenue) FILTER (WHERE first_purchase), 0) AS new_customer_revenue,
           count(*) FILTER (WHERE paid) AS paid_purchases,
           coalesce(sum(revenue) FILTER (WHERE paid), 0) AS paid_revenue
    FROM analysis.purchases GROUP BY 1),
monthly AS (
    SELECT month, coalesce(spend, 0) AS spend,
           coalesce(purchases, 0) AS purchases, coalesce(revenue, 0) AS revenue,
           coalesce(new_customers, 0) AS new_customers,
           coalesce(new_customer_revenue, 0) AS new_customer_revenue,
           coalesce(paid_purchases, 0) AS paid_purchases,
           coalesce(paid_revenue, 0) AS paid_revenue
    FROM spend FULL JOIN sales USING (month))
SELECT month,
       sum(spend) AS spend,
       sum(purchases) AS purchases,
       sum(revenue) AS revenue,
       sum(new_customers) AS new_customers,
       sum(new_customer_revenue) AS new_customer_revenue,
       sum(paid_purchases) AS paid_purchases,
       sum(paid_revenue) AS paid_revenue,
       sum(revenue) / nullif(sum(spend), 0) AS blended_roas,
       (sum(revenue) - sum(spend)) / nullif(sum(spend), 0) AS blended_roi,
       sum(paid_revenue) / nullif(sum(spend), 0) AS paid_roas,
       sum(spend) / nullif(sum(new_customers), 0) AS blended_cac
FROM monthly
GROUP BY GROUPING SETS ((month), ());

-- Acquisition cohorts: for each month customers first bought in, and each month after
-- it up to the data's last, how many of them bought, and what they spent. Months with
-- no purchases from the cohort are rows of zeros, so the cumulative columns run on.
-- `complete` is false for a month the data ends inside of.
CREATE VIEW analysis.cohorts AS
WITH sizes AS (
    SELECT cohort_month, count(*) AS cohort_size FROM analysis.customers GROUP BY 1),
grid AS (
    SELECT z.cohort_month, z.cohort_size, k AS months_since,
           (z.cohort_month + k * interval '1 month')::date AS month
    FROM sizes z
    CROSS JOIN analysis.period p
    CROSS JOIN LATERAL generate_series(
        0,
        ((extract(year FROM p.last_day) - extract(year FROM z.cohort_month)) * 12
         + extract(month FROM p.last_day) - extract(month FROM z.cohort_month))::integer
    ) AS k),
activity AS (
    SELECT c.cohort_month, date_trunc('month', p.session_date)::date AS month,
           count(DISTINCT p.user_pseudo_id) AS active_customers,
           count(*) AS purchases,
           sum(p.revenue) AS revenue
    FROM analysis.purchases p
    JOIN analysis.customers c USING (user_pseudo_id)
    GROUP BY 1, 2)
SELECT g.cohort_month, g.months_since, g.month, g.cohort_size,
       coalesce(a.active_customers, 0) AS active_customers,
       coalesce(a.purchases, 0) AS purchases,
       coalesce(a.revenue, 0) AS revenue,
       coalesce(a.active_customers, 0)::numeric / g.cohort_size AS retention,
       sum(coalesce(a.revenue, 0)) OVER w / g.cohort_size AS cumulative_revenue_per_customer,
       (sum(coalesce(a.purchases, 0)) OVER w)::numeric / g.cohort_size
           AS cumulative_purchases_per_customer,
       (g.month + interval '1 month' - interval '1 day')::date <= p.last_day AS complete
FROM grid g
CROSS JOIN analysis.period p
LEFT JOIN activity a USING (cohort_month, month)
WINDOW w AS (PARTITION BY g.cohort_month ORDER BY g.months_since);

-- Revenue per customer in the first `window_days` days from their first purchase
-- (that day included), by the channel group that acquired them, against the CAC paid
-- over the same acquisition dates. Only customers whose window ends by the data's last
-- day count, so every one is followed for the whole window; the spend and the new
-- customers in the CAC are taken over those same acquisition dates.
CREATE FUNCTION analysis.ltv_by_channel(window_days integer)
RETURNS TABLE (
    channel_group text, paid boolean, customers bigint, spend numeric, cac numeric,
    revenue_per_customer numeric, ltv_to_cac numeric
)
LANGUAGE sql STABLE
AS $$
    WITH p AS (
        SELECT last_day - window_days + 1 AS last_acquired FROM analysis.period),
    eligible AS (
        SELECT c.* FROM analysis.customers c, p WHERE c.acquired_on <= p.last_acquired),
    value AS (
        SELECT e.acquired_via, e.acquired_paid, e.user_pseudo_id, sum(x.revenue) AS revenue
        FROM eligible e
        JOIN analysis.purchases x USING (user_pseudo_id)
        WHERE x.session_date < e.acquired_on + window_days
        GROUP BY 1, 2, 3),
    spend AS (
        SELECT d.channel_group, sum(d.spend) AS spend
        FROM analysis.channel_daily d, p
        WHERE d.date <= p.last_acquired GROUP BY 1)
    SELECT v.acquired_via, bool_or(v.acquired_paid), count(*),
           s.spend,
           s.spend / count(*),
           sum(v.revenue) / count(*),
           (sum(v.revenue) / count(*)) / nullif(s.spend / count(*), 0)
    FROM value v LEFT JOIN spend s ON s.channel_group = v.acquired_via
    GROUP BY v.acquired_via, s.spend
    ORDER BY bool_or(v.acquired_paid) DESC, v.acquired_via
$$;
