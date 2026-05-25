-- RetailPulse: Stage 2 — Business Queries
-- Run in MySQL Workbench after Stage 1 ETL is complete
USE retailpulse;

-- ─────────────────────────────────────────────
-- Q1. Monthly revenue trend
-- Business: Are we growing or declining?
-- ─────────────────────────────────────────────
SELECT
    DATE_FORMAT(i.invoice_date, '%Y-%m')   AS month,
    ROUND(SUM(ii.quantity * ii.unit_price), 2) AS revenue,
    COUNT(DISTINCT i.invoice_no)            AS num_orders,
    COUNT(DISTINCT i.customer_id)           AS unique_customers
FROM invoices i
JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
GROUP BY month
ORDER BY month;


-- ─────────────────────────────────────────────
-- Q2. Top 10 customers by revenue
-- Business: Who are our most valuable customers?
-- ─────────────────────────────────────────────
SELECT
    i.customer_id,
    c.country,
    ROUND(SUM(ii.quantity * ii.unit_price), 2) AS total_revenue,
    COUNT(DISTINCT i.invoice_no)               AS total_orders
FROM invoices i
JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
JOIN customers c      ON i.customer_id = c.customer_id
GROUP BY i.customer_id, c.country
ORDER BY total_revenue DESC
LIMIT 10;


-- ─────────────────────────────────────────────
-- Q3. Revenue by country
-- Business: Which markets drive the most value?
-- ─────────────────────────────────────────────
SELECT
    c.country,
    ROUND(SUM(ii.quantity * ii.unit_price), 2) AS revenue,
    COUNT(DISTINCT i.customer_id)              AS customers,
    COUNT(DISTINCT i.invoice_no)               AS orders
FROM invoices i
JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
JOIN customers c      ON i.customer_id = c.customer_id
GROUP BY c.country
ORDER BY revenue DESC
LIMIT 15;


-- ─────────────────────────────────────────────
-- Q4. Top 10 best-selling products by quantity
-- Business: What should we always keep in stock?
-- ─────────────────────────────────────────────
SELECT
    ii.stock_code,
    p.description,
    SUM(ii.quantity)                           AS total_qty_sold,
    ROUND(SUM(ii.quantity * ii.unit_price), 2) AS total_revenue
FROM invoice_items ii
JOIN products p ON ii.stock_code = p.stock_code
GROUP BY ii.stock_code, p.description
ORDER BY total_qty_sold DESC
LIMIT 10;


-- ─────────────────────────────────────────────
-- Q5. Average order value (AOV) by month
-- Business: Are customers spending more per visit over time?
-- ─────────────────────────────────────────────
SELECT
    DATE_FORMAT(i.invoice_date, '%Y-%m')       AS month,
    ROUND(
        SUM(ii.quantity * ii.unit_price) /
        COUNT(DISTINCT i.invoice_no), 2
    )                                          AS avg_order_value
FROM invoices i
JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
GROUP BY month
ORDER BY month;


-- ─────────────────────────────────────────────
-- Q6. Customer purchase frequency distribution
-- Business: How many customers buy once vs repeatedly?
-- ─────────────────────────────────────────────
SELECT
    total_orders,
    COUNT(*) AS num_customers
FROM (
    SELECT
        customer_id,
        COUNT(DISTINCT invoice_no) AS total_orders
    FROM invoices
    GROUP BY customer_id
) order_counts
GROUP BY total_orders
ORDER BY total_orders;


-- ─────────────────────────────────────────────
-- Q7. Revenue by day of week
-- Business: Which days are busiest? Useful for staffing/campaigns.
-- ─────────────────────────────────────────────
SELECT
    DAYNAME(i.invoice_date)                    AS day_of_week,
    DAYOFWEEK(i.invoice_date)                  AS day_num,
    ROUND(SUM(ii.quantity * ii.unit_price), 2) AS revenue,
    COUNT(DISTINCT i.invoice_no)               AS orders
FROM invoices i
JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
GROUP BY day_of_week, day_num
ORDER BY day_num;


-- ─────────────────────────────────────────────
-- Q8. Monthly new vs returning customers
-- Business: Is our retention improving?
-- ─────────────────────────────────────────────
WITH first_purchase AS (
    SELECT
        customer_id,
        DATE_FORMAT(MIN(invoice_date), '%Y-%m') AS first_month
    FROM invoices
    GROUP BY customer_id
),
monthly_customers AS (
    SELECT
        DATE_FORMAT(i.invoice_date, '%Y-%m') AS month,
        i.customer_id,
        fp.first_month
    FROM invoices i
    JOIN first_purchase fp ON i.customer_id = fp.customer_id
    GROUP BY month, i.customer_id, fp.first_month
)
SELECT
    month,
    COUNT(CASE WHEN month = first_month THEN 1 END) AS new_customers,
    COUNT(CASE WHEN month > first_month  THEN 1 END) AS returning_customers
FROM monthly_customers
GROUP BY month
ORDER BY month;


-- ─────────────────────────────────────────────
-- Q9. RFM base table (feeds into ML stage)
-- Business: Segment customers by behavior
-- ─────────────────────────────────────────────
SELECT
    i.customer_id,
    DATEDIFF('2011-12-10', MAX(i.invoice_date))        AS recency_days,
    COUNT(DISTINCT i.invoice_no)                        AS frequency,
    ROUND(SUM(ii.quantity * ii.unit_price), 2)          AS monetary
FROM invoices i
JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
GROUP BY i.customer_id
ORDER BY monetary DESC;


-- ─────────────────────────────────────────────
-- Q10. Cohort retention (month 0 vs month 1)
-- Business: What % of new customers return next month?
-- ─────────────────────────────────────────────
WITH cohort AS (
    SELECT
        customer_id,
        DATE_FORMAT(MIN(invoice_date), '%Y-%m') AS cohort_month
    FROM invoices
    GROUP BY customer_id
),
activity AS (
    SELECT
        i.customer_id,
        DATE_FORMAT(i.invoice_date, '%Y-%m')    AS activity_month,
        c.cohort_month,
        PERIOD_DIFF(
            DATE_FORMAT(i.invoice_date, '%Y%m'),
            DATE_FORMAT(STR_TO_DATE(c.cohort_month, '%Y-%m'), '%Y%m')
        )                                        AS months_since_first
    FROM invoices i
    JOIN cohort c ON i.customer_id = c.customer_id
)
SELECT
    cohort_month,
    COUNT(DISTINCT CASE WHEN months_since_first = 0 THEN customer_id END) AS m0,
    COUNT(DISTINCT CASE WHEN months_since_first = 1 THEN customer_id END) AS m1,
    COUNT(DISTINCT CASE WHEN months_since_first = 2 THEN customer_id END) AS m2,
    COUNT(DISTINCT CASE WHEN months_since_first = 3 THEN customer_id END) AS m3
FROM activity
GROUP BY cohort_month
ORDER BY cohort_month;