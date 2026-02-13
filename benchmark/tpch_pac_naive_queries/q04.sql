WITH samples AS (
    -- 128 sample positions per order (privacy unit = orders.o_orderkey)
    SELECT o.o_orderkey AS pu_key, s.sample_id
    FROM orders o
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    -- per-sample count of orders satisfying the lineitem condition, grouped by sample and order priority
    SELECT
        s.sample_id,
        o.o_orderpriority,
        COUNT(*) AS cnt_order
    FROM samples s
    JOIN orders o ON o.o_orderkey = s.pu_key
    JOIN customer c ON c.c_custkey = o.o_custkey
    WHERE o.o_orderdate >= DATE '1993-07-01'
      AND o.o_orderdate < DATE '1993-10-01'
      AND EXISTS (
          SELECT 1 FROM lineitem l
          WHERE l.l_orderkey = o.o_orderkey
            AND l.l_commitdate < l.l_receiptdate
      )
    GROUP BY s.sample_id, o.o_orderpriority
)
SELECT
    o_orderpriority,
    pac_aggregate(array_agg(cnt_order ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS order_count
FROM per_sample
GROUP BY o_orderpriority
ORDER BY o_orderpriority;

