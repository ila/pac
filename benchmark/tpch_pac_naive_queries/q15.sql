WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    -- per-sample supplier revenue using samples derived from customer
    SELECT
        s.sample_id,
        l.l_suppkey AS supplier_no,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
        COUNT(*) AS cnt_order
    FROM samples s
    JOIN customer c ON c.c_custkey = s.pu_key
    JOIN orders o ON o.o_custkey = c.c_custkey
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    WHERE l.l_shipdate >= DATE '1996-01-01'
      AND l.l_shipdate < DATE '1996-04-01'
    GROUP BY s.sample_id, l.l_suppkey
),
per_supplier AS (
    -- PAC estimate per supplier by aggregating the per-sample revenues into arrays
    SELECT
        supplier_no,
        pac_aggregate(array_agg(total_revenue ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS pac_total_revenue
    FROM per_sample
    GROUP BY supplier_no
)
SELECT
    s.s_suppkey,
    s.s_name,
    s.s_address,
    s.s_phone,
    ps.pac_total_revenue AS total_revenue
FROM supplier s
JOIN per_supplier ps ON s.s_suppkey = ps.supplier_no
WHERE ps.pac_total_revenue = (
    SELECT MAX(pac_total_revenue) FROM per_supplier
)
ORDER BY s.s_suppkey;
