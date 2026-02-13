WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    -- per-sample aggregated volume by supplier nation, customer nation, and ship year
    SELECT
        s.sample_id,
        n1.n_name AS supp_nation,
        n2.n_name AS cust_nation,
        EXTRACT(year FROM l.l_shipdate) AS l_year,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS volume,
        COUNT(*) AS cnt_order
    FROM samples s
    JOIN customer c ON c.c_custkey = s.pu_key
    JOIN orders o ON o.o_custkey = c.c_custkey
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    JOIN supplier su ON su.s_suppkey = l.l_suppkey
    JOIN nation n1 ON su.s_nationkey = n1.n_nationkey
    JOIN nation n2 ON c.c_nationkey = n2.n_nationkey
    WHERE l.l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
      AND ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY') OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE'))
    GROUP BY s.sample_id, n1.n_name, n2.n_name, EXTRACT(year FROM l.l_shipdate)
)
SELECT
    supp_nation,
    cust_nation,
    l_year,
    pac_aggregate(array_agg(volume ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS revenue
FROM per_sample
GROUP BY supp_nation, cust_nation, l_year
ORDER BY supp_nation, cust_nation, l_year;
