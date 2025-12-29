WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    -- per-sample count of waiting orders per supplier name (join samples -> customer -> orders -> lineitem -> supplier -> nation)
    SELECT
        sm.sample_id,
        su.s_name,
        COUNT(*) AS numwait
    FROM samples sm
    JOIN customer c ON c.c_custkey = sm.pu_key
    JOIN orders o ON o.o_custkey = c.c_custkey
    JOIN lineitem l1 ON l1.l_orderkey = o.o_orderkey
    JOIN supplier su ON su.s_suppkey = l1.l_suppkey
    JOIN nation n ON su.s_nationkey = n.n_nationkey
    WHERE
        o.o_orderstatus = 'F'
        AND l1.l_receiptdate > l1.l_commitdate
        AND EXISTS (
            SELECT 1 FROM lineitem l2
            WHERE l2.l_orderkey = l1.l_orderkey
              AND l2.l_suppkey <> l1.l_suppkey
        )
        AND NOT EXISTS (
            SELECT 1 FROM lineitem l3
            WHERE l3.l_orderkey = l1.l_orderkey
              AND l3.l_suppkey <> l1.l_suppkey
              AND l3.l_receiptdate > l3.l_commitdate
        )
        AND n.n_name = 'SAUDI ARABIA'
    GROUP BY sm.sample_id, su.s_name
)
SELECT
    s_name,
    pac_aggregate(array_agg(numwait ORDER BY sample_id), array_agg(numwait ORDER BY sample_id), 1.0/128, 3) AS numwait
FROM per_sample
GROUP BY s_name
ORDER BY numwait DESC, s_name
LIMIT 100;
