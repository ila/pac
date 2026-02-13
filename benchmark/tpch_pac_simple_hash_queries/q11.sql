SELECT
    ps_partkey,
    pac_sum(hash(l_orderkey), ps_supplycost * ps_availqty) AS value
FROM
    partsupp,
    supplier,
    nation
WHERE
    ps_suppkey = s_suppkey
    AND s_nationkey = n_nationkey
    AND n_name = 'GERMANY'
GROUP BY
    ps_partkey
HAVING
    pac_sum(hash(l_orderkey), ps_supplycost * ps_availqty) > (
        SELECT
            pac_sum(hash(l_orderkey), ps_supplycost * ps_availqty) * 0.0001000000
        FROM
            partsupp,
            supplier,
            nation
        WHERE
            ps_suppkey = s_suppkey
            AND s_nationkey = n_nationkey
            AND n_name = 'GERMANY')
ORDER BY
    value DESC;
