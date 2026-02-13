WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    -- per-sample sums for promo and total volume for the month
    SELECT
        s.sample_id,
        100 * SUM(CASE WHEN p.p_type LIKE 'PROMO%' THEN l.l_extendedprice * (1 - l.l_discount) ELSE 0 END) AS promo_sum,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sum,
        COUNT(*) AS cnt_order
    FROM samples s
    JOIN customer c ON c.c_custkey = s.pu_key
    JOIN orders o ON o.o_custkey = c.c_custkey
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    JOIN part p ON l.l_partkey = p.p_partkey
    WHERE l.l_shipdate >= DATE '1995-09-01'
      AND l.l_shipdate < DATE '1995-10-01'
    GROUP BY s.sample_id
)
SELECT
    -- ratio of PAC estimates: numerator = PAC on per-sample promo_sum, denominator = PAC on per-sample total_sum
    (pac_aggregate(array_agg(promo_sum ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3)
     /
     NULLIF(pac_aggregate(array_agg(total_sum ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3), 0.0)
    ) AS promo_revenue
FROM per_sample;
