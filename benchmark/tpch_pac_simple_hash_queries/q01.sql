SELECT
    l_returnflag,
    l_linestatus,
    pac_sum(hash(l_orderkey), l_quantity) AS sum_qty,
    pac_sum(hash(l_orderkey), l_extendedprice) AS sum_base_price,
    pac_sum(hash(l_orderkey), l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    pac_sum(hash(l_orderkey), l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    pac_avg(hash(l_orderkey), l_quantity) AS avg_qty,
    pac_avg(hash(l_orderkey), l_extendedprice) AS avg_price,
    pac_avg(hash(l_orderkey), l_discount) AS avg_disc,
    pac_count(hash(l_orderkey)) AS count_order
FROM
    lineitem
WHERE
    l_shipdate <= CAST('1998-09-02' AS date)
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus;
