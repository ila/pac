WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    -- per-sample total volume and Brazil-specific volume for each order year
    SELECT
        s.sample_id,
        EXTRACT(year FROM o.o_orderdate) AS o_year,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_volume,
        SUM(CASE WHEN n2.n_name = 'BRAZIL' THEN l.l_extendedprice * (1 - l.l_discount) ELSE 0 END) AS brazil_volume,
        COUNT(*) AS cnt_order
    FROM samples s
    JOIN customer c ON c.c_custkey = s.pu_key
    JOIN orders o ON o.o_custkey = c.c_custkey
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    JOIN part p ON p.p_partkey = l.l_partkey
    JOIN supplier su ON su.s_suppkey = l.l_suppkey
    JOIN nation n2 ON su.s_nationkey = n2.n_nationkey
    JOIN nation n1 ON c.c_nationkey = n1.n_nationkey
    JOIN region r ON n1.n_regionkey = r.r_regionkey
    WHERE r.r_name = 'AMERICA'
      AND o.o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
      AND p.p_type = 'ECONOMY ANODIZED STEEL'
    GROUP BY s.sample_id, EXTRACT(year FROM o.o_orderdate)
)
SELECT
    o_year,
    -- ratio of PAC estimates: numerator = PAC on per-sample brazil_volume, denominator = PAC on per-sample total_volume
    (pac_aggregate(array_agg(brazil_volume ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3)
     /
     NULLIF(pac_aggregate(array_agg(total_volume ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3), 0.0)
    ) AS mkt_share
FROM per_sample
GROUP BY o_year
ORDER BY o_year;

