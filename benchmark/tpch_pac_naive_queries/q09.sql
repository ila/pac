WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    -- compute per-sample profit amount grouped by sample_id, nation, and order year
    SELECT
        s.sample_id,
        n.n_name AS nation,
        EXTRACT(year FROM o.o_orderdate) AS o_year,
        SUM(l.l_extendedprice * (1 - l.l_discount) - ps.ps_supplycost * l.l_quantity) AS amount,
        COUNT(*) AS cnt_order
    FROM samples s
    JOIN customer c ON c.c_custkey = s.pu_key
    JOIN orders o ON o.o_custkey = c.c_custkey
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    JOIN partsupp ps ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey
    JOIN part p ON p.p_partkey = l.l_partkey
    JOIN supplier su ON su.s_suppkey = l.l_suppkey
    JOIN nation n ON su.s_nationkey = n.n_nationkey
    WHERE p.p_name LIKE '%green%'
    GROUP BY s.sample_id, n.n_name, EXTRACT(year FROM o.o_orderdate)
)
SELECT
    nation,
    o_year,
    pac_aggregate(array_agg(amount ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS sum_profit
FROM per_sample
GROUP BY nation, o_year
ORDER BY nation, o_year DESC;
