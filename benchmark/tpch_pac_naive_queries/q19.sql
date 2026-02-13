WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    -- per-sample revenue using the original disjunction of part/lineitem predicates
    SELECT
        s.sample_id,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
        COUNT(*) AS cnt_order
    FROM samples s
    JOIN customer c ON c.c_custkey = s.pu_key
    JOIN orders o ON o.o_custkey = c.c_custkey
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    JOIN part p ON p.p_partkey = l.l_partkey
    WHERE (
        p.p_brand = 'Brand#12'
        AND p.p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
        AND l.l_quantity >= 1
        AND l.l_quantity <= 1 + 10
        AND p.p_size BETWEEN 1 AND 5
        AND l.l_shipmode IN ('AIR', 'AIR REG')
        AND l.l_shipinstruct = 'DELIVER IN PERSON'
    ) OR (
        p.p_brand = 'Brand#23'
        AND p.p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
        AND l.l_quantity >= 10
        AND l.l_quantity <= 10 + 10
        AND p.p_size BETWEEN 1 AND 10
        AND l.l_shipmode IN ('AIR', 'AIR REG')
        AND l.l_shipinstruct = 'DELIVER IN PERSON'
    ) OR (
        p.p_brand = 'Brand#34'
        AND p.p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
        AND l.l_quantity >= 20
        AND l.l_quantity <= 20 + 10
        AND p.p_size BETWEEN 1 AND 15
        AND l.l_shipmode IN ('AIR', 'AIR REG')
        AND l.l_shipinstruct = 'DELIVER IN PERSON'
    )
    GROUP BY s.sample_id
)
SELECT
    pac_aggregate(array_agg(revenue ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS revenue
FROM per_sample;
