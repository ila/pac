WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    -- per-sample revenue using original lineitem predicates
    SELECT
        s.sample_id,
        SUM(l.l_extendedprice * l.l_discount) AS revenue,
        COUNT(*) AS cnt_order
    FROM samples s
    JOIN customer c ON c.c_custkey = s.pu_key
    JOIN orders o ON o.o_custkey = c.c_custkey
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    WHERE l.l_shipdate >= DATE '1994-01-01'
      AND l.l_shipdate < DATE '1995-01-01'
      AND l.l_discount BETWEEN 0.05 AND 0.07
      AND l.l_quantity < 24
    GROUP BY s.sample_id
)
SELECT
    pac_aggregate(array_agg(revenue ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS revenue
FROM per_sample;
