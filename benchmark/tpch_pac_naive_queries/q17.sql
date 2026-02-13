WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    -- per-sample version of the original aggregate: join samples -> customer -> orders -> lineitem -> part
    SELECT
        s.sample_id,
        SUM(l.l_extendedprice) / 7.0 AS avg_yearly,
        COUNT(*) AS cnt_order
    FROM samples s
    JOIN customer c ON c.c_custkey = s.pu_key
    JOIN orders o ON o.o_custkey = c.c_custkey
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    JOIN part p ON p.p_partkey = l.l_partkey
    WHERE
        p.p_brand = 'Brand#23'
        AND p.p_container = 'MED BOX'
        AND l.l_quantity < (
            SELECT
                0.2 * avg(l_quantity)
            FROM
                lineitem
            WHERE
                l_partkey = p.p_partkey
        )
    GROUP BY s.sample_id
)
SELECT
    pac_aggregate(array_agg(avg_yearly ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS avg_yearly
FROM per_sample;

