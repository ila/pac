SELECT
    c_count,
    pac_count(hash(c_custkey), 1) AS custdist
FROM (
    SELECT
        c_custkey,
        pac_count(hash(o_orderkey), o_orderkey)
    FROM
        customer
    LEFT OUTER JOIN orders ON c_custkey = o_custkey
    AND o_comment NOT LIKE '%special%requests%'
GROUP BY
    c_custkey) AS c_orders (c_custkey,
        c_count)
GROUP BY
    c_count
ORDER BY
    custdist DESC,
    c_count DESC;
