SELECT
    c_count,
    count(*) AS custdist
FROM (
         SELECT
             customer.c_custkey,
             pac_count(hash(orders.o_orderkey), o_orderkey) AS c_count
         FROM
             customer
                 LEFT OUTER JOIN orders
                                 ON customer.c_custkey = orders.o_custkey
                                     AND orders.o_comment NOT LIKE '%special%requests%'
         GROUP BY
             customer.c_custkey
     ) AS c_orders (c_custkey, c_count)
GROUP BY
    c_count
ORDER BY
    custdist DESC,
    c_count DESC;
