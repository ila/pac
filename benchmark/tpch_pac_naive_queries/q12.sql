WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    -- per-sample high/low line counts grouped by sample and shipmode
    SELECT
        s.sample_id,
        l.l_shipmode,
        SUM(CASE WHEN o.o_orderpriority = '1-URGENT' OR o.o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
        SUM(CASE WHEN o.o_orderpriority <> '1-URGENT' AND o.o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count,
        COUNT(*) AS cnt_order
    FROM samples s
    JOIN customer c ON c.c_custkey = s.pu_key
    JOIN orders o ON o.o_custkey = c.c_custkey
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    WHERE l.l_shipmode IN ('MAIL', 'SHIP')
      AND l.l_commitdate < l.l_receiptdate
      AND l.l_shipdate < l.l_commitdate
      AND l.l_receiptdate >= DATE '1994-01-01'
      AND l.l_receiptdate < DATE '1995-01-01'
    GROUP BY s.sample_id, l.l_shipmode
)
SELECT
    l_shipmode,
    pac_aggregate(array_agg(high_line_count ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS high_line_count,
    pac_aggregate(array_agg(low_line_count ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS low_line_count
FROM per_sample
GROUP BY l_shipmode
ORDER BY l_shipmode;
