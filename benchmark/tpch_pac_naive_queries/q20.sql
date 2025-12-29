WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
inner_matches AS (
    -- inner query: join samples -> customer -> orders -> lineitem -> part -> partsupp -> supplier -> nation
    -- produce one row per sample_id and matching partsupp supplier
    SELECT
        sm.sample_id,
        ps.ps_suppkey AS supplier_no
    FROM samples sm
    JOIN customer c ON c.c_custkey = sm.pu_key
    JOIN orders o ON o.o_custkey = c.c_custkey
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    JOIN part p ON p.p_partkey = l.l_partkey
    JOIN partsupp ps ON ps.ps_partkey = p.p_partkey AND ps.ps_suppkey = l.l_suppkey
    JOIN supplier s2 ON s2.s_suppkey = ps.ps_suppkey
    JOIN nation n ON s2.s_nationkey = n.n_nationkey
    WHERE p.p_name LIKE 'forest%'
      AND ps.ps_availqty > (
          SELECT 0.5 * SUM(l2.l_quantity)
          FROM lineitem l2
          WHERE l2.l_partkey = ps.ps_partkey
            AND l2.l_suppkey = ps.ps_suppkey
            AND l2.l_shipdate >= DATE '1994-01-01'
            AND l2.l_shipdate < DATE '1995-01-01'
      )
      AND n.n_name = 'CANADA'
),
per_sample AS (
    -- aggregate per sample: count matches per sample_id and supplier
    SELECT
        sample_id,
        supplier_no,
        COUNT(*) AS match_count
    FROM inner_matches
    GROUP BY sample_id, supplier_no
),
per_supplier AS (
    -- compute PAC estimate (noisy count) per supplier by aggregating per-sample match counts
    SELECT
        supplier_no,
        pac_aggregate(array_agg(match_count ORDER BY sample_id), array_agg(match_count ORDER BY sample_id), 1.0/128, 3) AS pac_match_score
    FROM per_sample
    GROUP BY supplier_no
)
SELECT
    s.s_name,
    s.s_address,
    ps.pac_match_score AS match_score
FROM supplier s
JOIN per_supplier ps ON s.s_suppkey = ps.supplier_no
ORDER BY s.s_name;
