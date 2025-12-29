WITH samples AS (
    -- 128 sample positions per privacy unit (customer.c_custkey used as privacy unit key)
    -- include each (customer, sample_id) with probability 0.5
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
    WHERE random() < 0.5
),
per_sample AS (
    -- compute per-sample aggregates by joining samples -> orders -> lineitem
    SELECT
        s.sample_id,
        l.l_returnflag,
        l.l_linestatus,
        SUM(l.l_quantity) AS sum_qty,
        SUM(l.l_extendedprice) AS sum_base_price,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS sum_disc_price,
        SUM(l.l_extendedprice * (1 - l.l_discount) * (1 + l.l_tax)) AS sum_charge,
        AVG(l.l_quantity) AS avg_qty,
        AVG(l.l_extendedprice) AS avg_price,
        AVG(l.l_discount) AS avg_disc,
        COUNT(*) AS cnt_order
    FROM samples s
    JOIN orders o ON o.o_custkey = s.pu_key
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    WHERE l.l_shipdate <= DATE '1998-09-02'
    GROUP BY s.sample_id, l.l_returnflag, l.l_linestatus
)
SELECT
    l_returnflag,
    l_linestatus,
    pac_aggregate(array_agg(sum_qty ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS sum_qty,
    pac_aggregate(array_agg(sum_base_price ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS sum_base_price,
    pac_aggregate(array_agg(sum_disc_price ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS sum_disc_price,
    pac_aggregate(array_agg(sum_charge ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS sum_charge,
    pac_aggregate(array_agg(avg_qty ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS avg_qty,
    pac_aggregate(array_agg(avg_price ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS avg_price,
    pac_aggregate(array_agg(avg_disc ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS avg_disc,
    pac_aggregate(array_agg(cnt_order ORDER BY sample_id), array_agg(cnt_order ORDER BY sample_id), 1.0/128, 3) AS count_order
FROM per_sample
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;
