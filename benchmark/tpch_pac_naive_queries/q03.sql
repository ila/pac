WITH samples AS (
    -- sample positions drawn from customers (no filtering; include all customers x 1..128)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    -- per-sample revenue per order (group keys: l_orderkey, o_orderdate, o_shippriority)
    SELECT
        s.sample_id,
        l.l_orderkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
        o.o_orderdate,
        o.o_shippriority,
        COUNT(*) AS cnt_order
    FROM samples s
    JOIN customer c ON c.c_custkey = s.pu_key
    JOIN orders o ON o.o_custkey = c.c_custkey
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    WHERE o.o_orderdate < DATE '1995-03-15'
      AND l.l_shipdate > DATE '1995-03-15'
    GROUP BY s.sample_id, l.l_orderkey, o.o_orderdate, o.o_shippriority
)
SELECT
    l_orderkey,
    pac_aggregate(array_agg(revenue ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS revenue,
    o_orderdate,
    o_shippriority
FROM per_sample
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10;
