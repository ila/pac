WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    -- per-sample revenue for each nation name (n_name)
    SELECT
        s.sample_id,
        n.n_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
        COUNT(*) AS cnt_order
    FROM samples s
    JOIN customer c ON c.c_custkey = s.pu_key
    JOIN orders o ON o.o_custkey = c.c_custkey
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    JOIN supplier su ON l.l_suppkey = su.s_suppkey
    JOIN nation n ON su.s_nationkey = n.n_nationkey
    JOIN region r ON n.n_regionkey = r.r_regionkey
    WHERE r.r_name = 'ASIA'
      AND o.o_orderdate >= DATE '1994-01-01'
      AND o.o_orderdate < DATE '1995-01-01'
    GROUP BY s.sample_id, n.n_name
)
SELECT
    n_name,
    pac_aggregate(array_agg(revenue ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS revenue
FROM per_sample
GROUP BY n_name
ORDER BY revenue DESC;
